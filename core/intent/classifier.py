"""意图分类小模型：只在「规则判不准 / 复杂·多意图」时兜底复核。

为什么不做成「每句话都过一遍模型」：会话首响目标是 1.2s
（``config/constants.TARGET_VOICE_RESPONSE_MS``），每次输入都多一次模型往返会平白
抬高首响；而控制指令、闲聊、单一提问这些**最常见的输入**规则就能判准。因此模型只在
两类场景被调用（见 ``IntentAnalyzer._needs_model``）：

* 规则判为 ``UNKNOWN``（关键词与疑问特征都没命中）；
* 规则判为复杂 / 多意图（需要拆子问题、判定主意图到底是哪个）。

这两类本来就要走 LLM 答疑（秒级），多花约 200~500ms 做分类完全在预算内。

模型调用失败 / 超时 / 返回不合法一律返回 ``None``，由调用方回退到规则结果 ——
**分类可以不准，但绝不能拖慢或阻断答疑**。
"""
from __future__ import annotations

import asyncio
from typing import Any

from config.constants import IntentType
from config.logging import get_logger
from config.settings import settings
from models.base import BaseLLMClient
from models.llm import OpenAICompatLLM, get_llm

log = get_logger("core.intent.classifier")

# 严格 JSON 输出：字段少、指令前置，让「小模型」也能稳定照做。
# 注意 sub_questions 要求**照原话拆**（不改写、不加解释）：这份清单会原样喂给
# TeacherAgent 当任务列表，模型一旦"润色"就可能把学生的原意改掉。
_PROMPT = """你是意图分类器，只做分类，不要回答学生的问题。

判断下面这句话里有几个诉求，并给出主意图。只输出 JSON：
{{"intent": "control|chitchat|image_question|text_question",
 "multi": true/false,
 "complex": true/false,
 "sub_questions": ["子问题1", "子问题2"]}}

判定规则：
1. 一句话里有两个及以上**不同诉求**（如「这道题怎么解，另外帮我打开麦克风」）时 multi=true，
   并把每个诉求**按原文顺序**拆进 sub_questions：控制指令也要拆出来，照原话写，
   不要改写、不要合并、不要加解释；
2. 只是一个诉求、只是带了多个条件时 multi=false，sub_questions 放这一个诉求；
3. 需要分步推理的复杂题（多步证明/推导、多条件求解）complex=true；
4. intent 取最主要的那个诉求：设备或界面指令=control，打招呼寒暄=chitchat，
   看图解题=image_question，其余提问=text_question。

学生输入：{text}
"""

# 模型只允许在这些意图里选；``unknown`` 不作为模型输出（判不出来时应返回空 JSON，
# 由调用方保留规则结果，而不是让模型把「判不出」包装成一个新意图）。
_INTENT_BY_NAME: dict[str, IntentType] = {
    IntentType.CONTROL.value: IntentType.CONTROL,
    IntentType.CHITCHAT.value: IntentType.CHITCHAT,
    IntentType.IMAGE_QUESTION.value: IntentType.IMAGE_QUESTION,
    IntentType.TEXT_QUESTION.value: IntentType.TEXT_QUESTION,
    IntentType.VOICE_QUESTION.value: IntentType.VOICE_QUESTION,
}


class IntentLlmClassifier:
    """小模型意图分类器（严格 JSON + 超时/异常一律降级）。"""

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        *,
        model: str = "",
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._llm = llm or self._build_llm(model)
        self.timeout = float(
            timeout if timeout is not None else settings.intent_llm_timeout_seconds
        )
        self.max_tokens = int(
            max_tokens if max_tokens is not None else settings.intent_llm_max_tokens
        )

    @staticmethod
    def _build_llm(model: str) -> BaseLLMClient:
        """构造分类用客户端。

        只有「模型 / 地址 / 密钥」三者都与主 LLM 完全相同时才复用主客户端（零新增基建）；
        任一不同就单独建一个 —— 典型场景是意图识别走**独立密钥**（单独计费/限流，
        不占用答疑主链路的额度）或独立小模型。
        """
        name = (model or settings.intent_llm_model or settings.llm_model or "").strip()
        base = (settings.intent_llm_api_base or settings.llm_api_base or "").strip()
        key = (settings.intent_llm_api_key or settings.llm_api_key or "").strip()
        if not key:
            log.warning("未配置意图模型密钥，回退主 LLM 分类")
            return get_llm()
        same_as_main = (
            name == (settings.llm_model or "").strip()
            and base.rstrip("/") == (settings.llm_api_base or "").rstrip("/")
            and key == (settings.llm_api_key or "").strip()
        )
        if same_as_main:
            return get_llm()
        log.info("意图分类使用独立配置：model={} base={}", name, base or "(默认)")
        return OpenAICompatLLM(
            api_base=base,
            api_key=key,
            model=name,
            timeout=settings.intent_llm_timeout_seconds,
        )

    async def classify(self, text: str, *, base: IntentType | None = None) -> dict[str, Any] | None:
        """分类一段输入；返回归一化结果，失败/超时/不合法返回 ``None``。

        ``base`` 只是给调用方留的上下文口子（当前不参与提示词），便于将来做
        「规则说 A、模型说 B」的对比统计。
        """
        text = (text or "").strip()
        if not text:
            return None
        messages = [{"role": "user", "content": _PROMPT.format(text=text[:500])}]
        try:
            data = await asyncio.wait_for(
                self._llm.chat_json(messages, temperature=0.0, max_tokens=self.max_tokens),
                timeout=self.timeout,
            )
        except Exception as err:  # noqa: BLE001 - 分类失败绝不能影响答疑
            log.warning("意图分类模型调用失败（回退规则判定）：{}", err)
            return None
        return self._normalize(data)

    @staticmethod
    def _normalize(data: Any) -> dict[str, Any] | None:
        """校验模型输出：意图名不认识就丢弃，但子问题清单仍可用。"""
        if not isinstance(data, dict) or not data:
            return None
        intent = _INTENT_BY_NAME.get(str(data.get("intent") or "").strip().lower())
        subs = [str(s).strip() for s in (data.get("sub_questions") or []) if str(s).strip()]
        if intent is None and not subs:
            return None          # 没有任何可用信息：让调用方保留规则结果
        return {
            "intent": intent,
            "multi": bool(data.get("multi")),
            "complex": bool(data.get("complex")),
            "sub_questions": subs,
        }
