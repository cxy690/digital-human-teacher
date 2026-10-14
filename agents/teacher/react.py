"""ReAct 闭环：把答疑从「一次生成」升级为「思考 -> 行动 -> 观察 -> 再思考」。

加装之前的问题
--------------
答疑是**单向流水**：``generate`` 一次生成 JSON，``validate`` 只做格式校验，
``solve`` 只做一次事后验算。模型一旦算错或记错知识点，链路里没有任何一步
会让它「回头看自己的结论」——错误只会在白板上呈现给学生。

闭环长什么样
------------
::

    thought: "这题是方程，我先解出来再写讲解。"
    action : math_solve({"equation": "3x+5=20"})        <- 模型自己决定
    observe: "3x+5=20 的解：x=5"
    thought: "再验算一次这个答案。"
    action : math_verify({"equation": "3x+5=20", "answer": "x=5"})
    observe: "结论：x=5 满足 3x+5=20。"
    answer : {...最终 JSON 契约...}                      <- 不再调工具，收尾

几个刻意的设计
--------------
* **有界**：最多 ``agent_react_max_steps`` 轮「思考 + 工具」，工具总数另有
  ``agent_tool_max_calls`` 上限；步数用尽时会再要一次 ``tool_choice="none"``
  的收尾调用，逼模型给出答复（比在提示词里写「不要再调工具」可靠）。
* **可降级**：模型不支持工具、超时、异常，一律返回空文本 + 结束原因，
  由调用方回退到加装前的单次生成链路 —— 闭环是增强，不是新的故障点。
* **可回溯**：每一步的思考 / 工具 / 观察都进 :class:`ReactResult`，
  由调用方写进 ``turn.meta["react_trace"]``，出问题能逐帧回放。
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from agents.shared.toolkit import AgentToolkit
from config.logging import get_logger
from config.settings import settings
from models.base import BaseLLMClient, ChatMessages, ToolCall

log = get_logger("agents.teacher.react")

#: 结束原因：模型直接给了答复（正常收敛）
STOP_ANSWER = "answer"
#: 结束原因：步数用尽，收尾调用拿到的答复
STOP_MAX_STEPS = "max_steps"
#: 结束原因：模型不支持工具调用，退化为一次普通生成
STOP_NO_TOOLS = "no_tool_support"
#: 结束原因：超时 / 异常（调用方应回退到老链路）
STOP_TIMEOUT = "timeout"
STOP_ERROR = "error"


@dataclass
class ReactStep:
    """一轮「思考 + 行动 + 观察」。"""

    index: int
    thought: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought[:300],
            "calls": list(self.calls),
            "observations": [obs[:300] for obs in self.observations],
        }


@dataclass
class ReactResult:
    """一次闭环执行的产物。"""

    text: str = ""
    steps: list[ReactStep] = field(default_factory=list)
    stopped: str = STOP_ANSWER
    elapsed_ms: float = 0.0

    @property
    def used_tools(self) -> bool:
        return any(step.calls for step in self.steps)

    @property
    def tool_count(self) -> int:
        return sum(len(step.calls) for step in self.steps)

    def to_meta(self) -> dict[str, Any]:
        """写进 ``turn.meta`` 的可序列化快照（不含完整文本，避免 meta 过大）。"""
        return {
            "steps": len(self.steps),
            "tool_calls": self.tool_count,
            "stopped": self.stopped,
            "elapsed_ms": self.elapsed_ms,
            "trace": [step.to_dict() for step in self.steps],
        }


def _assistant_message(content: str, calls: list[ToolCall]) -> dict[str, Any]:
    """把模型的工具调用意图回填成 assistant 消息（原样保留服务端字段，缺的补默认值）。"""
    items: list[dict[str, Any]] = []
    for call in calls:
        raw = dict(call.raw or {})
        raw.setdefault("id", call.id)
        raw.setdefault("type", "function")
        fn = dict(raw.get("function") or {})
        fn.setdefault("name", call.name)
        fn.setdefault("arguments", json.dumps(call.arguments, ensure_ascii=False))
        raw["function"] = fn
        items.append(raw)
    return {"role": "assistant", "content": content or "", "tool_calls": items}


class ReActLoop:
    """把「模型 + 工具」跑成闭环的执行器。"""

    def __init__(
        self,
        llm: BaseLLMClient,
        toolkit: AgentToolkit | None,
        *,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.llm = llm
        self.toolkit = toolkit
        self.max_steps = int(
            max_steps if max_steps is not None else settings.agent_react_max_steps
        )
        self.timeout = float(
            timeout if timeout is not None else settings.agent_react_timeout_seconds
        )

    async def run(
        self,
        messages: ChatMessages,
        *,
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ) -> ReactResult:
        """执行闭环；返回最终答复文本与全部步骤。任何异常都收敛成结果对象。

        ``max_tokens=None`` 时取 ``ANSWER_MAX_TOKENS``（默认 1800）。这里原本写死 900 ——
        而答疑契约一次要产出「400 字话术 + 4~6 步 + 知识点 + 易错点 + 追问」，
        实测正好卡在 900 边界上：**截断不是「话短一点」，而是 JSON 不闭合 ->
        整块降级 -> 白板全空、步骤全丢**（L4 评测在云端实测到，规则分 100→35）。
        详见 ``config/settings.py`` 的 ``answer_max_tokens`` 注释。
        """
        if max_tokens is None:
            max_tokens = int(settings.answer_max_tokens)
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._run(messages, temperature=temperature, max_tokens=max_tokens),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            log.warning("ReAct 闭环超时（{}s），回退单次生成", self.timeout)
            result = ReactResult(stopped=STOP_TIMEOUT)
        except Exception as err:  # noqa: BLE001 - 闭环失败必须能回退
            log.warning("ReAct 闭环异常，回退单次生成：{}", err)
            result = ReactResult(stopped=STOP_ERROR)
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    # ---------------------------------------------------------------- 内部
    async def _run(
        self,
        messages: ChatMessages,
        *,
        temperature: float,
        max_tokens: int,
    ) -> ReactResult:
        tools = self.toolkit.schemas() if self.toolkit is not None else []
        if not tools or not self.llm.supports_tools:
            # 模型不支持工具（或工具开关关闭）：退化成一次普通生成，
            # 行为与加装闭环之前完全一致
            text = await self.llm.chat(
                messages, temperature=temperature, max_tokens=max_tokens, json_mode=True
            )
            return ReactResult(text=text, stopped=STOP_NO_TOOLS)

        convo: ChatMessages = list(messages)
        steps: list[ReactStep] = []
        last_content = ""
        for index in range(self.max_steps):
            reply = await self.llm.chat_with_tools(
                convo,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                # 第 1 轮不强制 JSON：这一轮要留给模型自由决定「调哪些工具」；
                # 从第 2 轮起它已经拿到观察结果、该给最终答复了 —— 而最终答复必须是
                # 合法 JSON 契约（实测不带 response_format 时长讲解偶发结构损坏，
                # 见 models/llm.py 的 chat_with_tools 说明）。
                json_mode=index > 0,
            )
            last_content = reply.content or last_content
            if not reply.wants_tools:
                return ReactResult(text=reply.content or "", steps=steps, stopped=STOP_ANSWER)

            step = ReactStep(index=index, thought=reply.content or "")
            convo.append(_assistant_message(reply.content or "", reply.tool_calls))
            for call in reply.tool_calls:
                observation = await self.toolkit.call(call.name, call.arguments)  # type: ignore[union-attr]
                step.calls.append(call.to_dict())
                step.observations.append(observation)
                convo.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": observation,
                    }
                )
            steps.append(step)
            log.info(
                "ReAct 第 {} 轮：调用 {} 个工具（{}）",
                index + 1, len(step.calls), "、".join(c["name"] for c in step.calls),
            )

        # 步数用尽：禁止再调工具，逼模型用已有观察给出答复（并强制合法 JSON）
        final = await self.llm.chat_with_tools(
            convo,
            tools=tools,
            tool_choice="none",
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        text = final.content or last_content
        return ReactResult(text=text, steps=steps, stopped=STOP_MAX_STEPS)
