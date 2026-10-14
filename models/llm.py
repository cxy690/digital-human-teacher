"""LLM 客户端：默认 OpenAI 兼容协议；无密钥时自动降级为 Mock，保证开箱可跑。"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, AsyncIterator

import httpx

from config.logging import get_logger
from config.settings import settings
from models.base import BaseLLMClient, ChatMessages, ToolCall, ToolChatResult

log = get_logger("models.llm")


def parse_tool_calls(raw: Any) -> list[ToolCall]:
    """把服务端返回的 ``message.tool_calls`` 归一化为 :class:`ToolCall` 列表。

    ``arguments`` 在 OpenAI 协议里是**JSON 字符串**，这里解析成 dict；
    解析失败（模型偶尔会吐出非 JSON）时保留原始串放进 ``arguments={"__raw__": ...}``，
    交给工具层报错而不是在这里抛异常 —— 一次工具调用不合法不该打断整节课。
    """
    calls: list[ToolCall] = []
    if not isinstance(raw, list):
        return calls
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        raw_args = fn.get("arguments")
        arguments: dict[str, Any] = {}
        if isinstance(raw_args, dict):
            arguments = raw_args
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
                arguments = parsed if isinstance(parsed, dict) else {"__value__": parsed}
            except json.JSONDecodeError:
                arguments = {"__raw__": raw_args}
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{index}"),
                name=name,
                arguments=arguments,
                raw=item,
            )
        )
    return calls


def _parse_extra_body(raw: str) -> dict[str, Any]:
    """把 ``LLM_EXTRA_BODY``（JSON 字符串）解析成字典；非法内容忽略并告警。

    用途：给 OpenAI 兼容接口传**厂商专有参数**（如 DeepSeek V4 的
    ``{"thinking": {"type": "disabled"}}``）。写错了只告警、不抛异常——
    绝不能因为一个可选配置让整个答疑链路起不来。
    """
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        log.warning("LLM_EXTRA_BODY 不是合法 JSON，已忽略：{} ({})", text, err)
        return {}
    if not isinstance(data, dict):
        log.warning("LLM_EXTRA_BODY 必须是 JSON 对象，已忽略：{}", text)
        return {}
    return data


class OpenAICompatLLM(BaseLLMClient):
    """OpenAI / 通义 / 智谱等兼容 ``/chat/completions`` 的客户端。"""

    supports_tools = True
    #: 是否支持 ``response_format={"type": "json_object"}``；
    #: 首次被厂商 4xx 拒绝后置 False（类级），后续不再尝试（见 chat）
    supports_json_mode = True

    def __init__(self, api_base: str, api_key: str, model: str, timeout: float = 30.0) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.extra_body = _parse_extra_body(settings.llm_extra_body)

    def _payload(
        self,
        messages: ChatMessages,
        *,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            payload["stream"] = True
        # 厂商专有参数（见 _parse_extra_body / settings.llm_extra_body）
        if self.extra_body:
            payload.update(self.extra_body)
        return payload

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def chat(
        self,
        messages: ChatMessages,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens)
        if json_mode and type(self).supports_json_mode:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions", json=payload, headers=self._headers()
            )
            if json_mode and resp.status_code in (400, 404, 415, 422):
                # 该厂商 / 该模型不认 response_format：记住并退回普通请求，
                # 免得每次长输出都白撞一次 4xx（答疑链路对延迟敏感）
                log.warning(
                    "LLM 不支持 response_format（{}），后续停用 JSON 模式",
                    resp.status_code,
                )
                type(self).supports_json_mode = False
                return await self.chat(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

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
        """带工具（function calling）的对话补全。

        ``tools`` 用 OpenAI 的 ``{"type": "function", "function": {...}}`` 结构；
        ``tool_choice="none"`` 用于「步数用尽、逼模型给最终答复」的那一次收尾调用
        （比在提示词里写「不要再调工具了」可靠得多）。

        ``json_mode``：强制最终答复为合法 JSON。**ReAct 闭环必须带上它** ——
        闭环的最后一句同样是上千 token 的 JSON 契约，模型偶发写坏结构
        （实测云端「原文 1200 字」的损坏），而这条路径走的是本方法、不是 ``chat``。
        """
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens)
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
        if json_mode and type(self).supports_json_mode:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions", json=payload, headers=self._headers()
            )
            if json_mode and resp.status_code in (400, 404, 415, 422):
                # 该厂商不认「工具 + JSON 模式」的组合：记住并退回（工具照旧保留）
                log.warning(
                    "LLM 不支持 tools+response_format（{}），后续停用 JSON 模式",
                    resp.status_code,
                )
                type(self).supports_json_mode = False
                return await self.chat_with_tools(
                    messages, tools=tools, tool_choice=tool_choice,
                    temperature=temperature, max_tokens=max_tokens,
                )
            resp.raise_for_status()
            data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return ToolChatResult(
            content=str(message.get("content") or ""),
            tool_calls=parse_tool_calls(message.get("tool_calls")),
            finish_reason=str(choice.get("finish_reason") or ""),
        )

    async def chat_stream(
        self,
        messages: ChatMessages,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """SSE 流式对话补全：逐段产出 ``choices[0].delta.content`` 文本增量。"""
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.api_base}/chat/completions", json=payload, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if not chunk or chunk == "[DONE]":
                        if chunk == "[DONE]":
                            break
                        continue
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    delta = (data.get("choices") or [{}])[0].get("delta") or {}
                    text = delta.get("content")
                    if text:
                        yield text


class Qwen3LLM(OpenAICompatLLM):
    """Qwen3 私有化部署适配器（立项方案 3.4 指定）。

    Qwen3 经 vLLM / SGLang 部署时兼容 OpenAI ``/chat/completions``，本类在 payload 中
    附带 Qwen3 特有的 ``chat_template_kwargs.enable_thinking`` 开关（直播场景关闭以降延迟）。
    """

    def __init__(self, *args: Any, enable_thinking: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.enable_thinking = enable_thinking

    def _payload(
        self,
        messages: ChatMessages,
        *,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages, temperature=temperature, max_tokens=max_tokens, stream=stream
        )
        payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        return payload


class MockLLM(BaseLLMClient):
    """离线 Mock：用规则拼装一个像样的答疑回复，便于无网络环境下联调与自测。

    ``tool_script`` 用于**离线验证 ReAct 闭环**（``agents/teacher/react.py``）：
    传入一串 ``(工具名, 参数字典)``，前几轮 ``chat_with_tools`` 会按顺序把它们
    作为 ``tool_calls`` 返回（模拟「模型决定先调工具」），脚本用尽后回到普通文本答复。
    不传脚本时行为与加装工具之前**完全一致**（一次返回文本、无工具调用），
    因此既有单测与离线演示的判定不受影响。
    """

    is_mock = True
    supports_tools = True

    def __init__(self, tool_script: list[tuple[str, dict[str, Any]]] | None = None) -> None:
        self._tool_script: list[tuple[str, dict[str, Any]]] = list(tool_script or [])

    @staticmethod
    def _user_text(messages: ChatMessages) -> str:
        """取最后一条 user 消息的纯文本（多模态 content 列表也兼容）。"""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                return " ".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
            return str(content)
        return ""

    async def chat(
        self,
        messages: ChatMessages,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,          # Mock 不需要（离线演示本来就是固定文本）
    ) -> str:
        topic = self._user_text(messages).strip()[:40] or "这道题"
        return (
            f"我们先看“{topic}”。第一步，提取题目已知条件；"
            "第二步，找到对应的知识点；第三步，代入公式计算。"
            "你可以先自己试算结果，我再帮你检查（当前为离线演示回复）。"
        )

    async def chat_with_tools(
        self,
        messages: ChatMessages,
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,          # Mock 不需要（离线演示回复本来就是纯文本）
    ) -> ToolChatResult:
        """按 ``tool_script`` 依次吐出工具调用；脚本用尽（或禁用工具）时给文本答复。"""
        tool_names = {
            str((tool.get("function") or {}).get("name") or "")
            for tool in tools
            if isinstance(tool, dict)
        }
        if tool_choice != "none" and self._tool_script:
            name, arguments = self._tool_script[0]
            # 工具名不在本次可用集合里就跳过，避免造出调用不到的工具名
            if name in tool_names:
                self._tool_script.pop(0)
                return ToolChatResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"mock_call_{len(self._tool_script)}",
                            name=name,
                            arguments=dict(arguments),
                            raw={
                                "id": f"mock_call_{len(self._tool_script)}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments, ensure_ascii=False),
                                },
                            },
                        )
                    ],
                    finish_reason="tool_calls",
                )
            self._tool_script.pop(0)
        return ToolChatResult(content=await self.chat(messages, temperature=temperature, max_tokens=max_tokens))


@lru_cache
def get_llm() -> BaseLLMClient:
    """按 ``llm_provider`` 返回 LLM 客户端；缺密钥时降级为 Mock。"""
    provider = (settings.llm_provider or "openai").lower()
    if not settings.llm_api_key:
        log.warning("no llm_api_key configured, fallback to MockLLM")
        return MockLLM()
    if provider == "qwen3":
        log.info("use Qwen3 (private deployment): {}", settings.llm_model)
        return Qwen3LLM(
            api_base=settings.llm_api_base,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            enable_thinking=settings.llm_enable_thinking,
        )
    log.info("use OpenAI-compatible LLM: {}", settings.llm_model)
    return OpenAICompatLLM(
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )
