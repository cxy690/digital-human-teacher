# -*- coding: utf-8 -*-
"""提示词构建 + LLM 生成（P1，链路第 7 步）。

约束：仅当 policy 判为 ``rag_grounded`` 时才走到这里；fixed/refuse/template 不进生成模型。

- :class:`LLMBackend` 抽象一个 ``complete(prompt) -> str``；
  提供 :class:`OllamaBackend`（本地 qwen2.5:7b，走其 REST，标准库 urllib，零新依赖）与
  :class:`DashScopeBackend`（OpenAI 兼容端点，api key 只从环境变量读）。
- Prompt 用团队模板 ``资料: {text}\n\n问题: {query}\n请基于资料回答:``，
  并增强要求**每个论断用 [kp_id] 标注**支撑它的资料。
- 生成后做**引用校验**：答案里出现的每个 ``[x]`` 必须属于本轮证据集的 kp_id；
  ``require_in_evidence`` 时还须至少有一个引用。失败重试一次，再失败降级为拒答话术。
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence

from . import config as _cfg

_CITATION = re.compile(r"\[([^\]]+)\]")

# 引用校验连续失败时的降级话术（无 LLM key 也始终可用）。
REFUSE_FALLBACK = "抱歉，我暂时没有足够的可靠资料来准确回答这个问题。"


class LLMBackend(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass
class Evidence:
    kp_id: str
    text: str


@dataclass
class GenerationResult:
    answer: str
    ok: bool              # 引用校验是否通过
    degraded: bool = False  # True 表示降级为拒答话术
    attempts: int = 1


class OllamaBackend:
    """本地 Ollama；用标准库 urllib 调 /api/generate，不引额外依赖。"""

    def __init__(self, gen_cfg: _cfg.GenerationConfig):
        self.base_url = gen_cfg.ollama.base_url.rstrip("/")
        self.model = gen_cfg.ollama.model
        self.temperature = gen_cfg.temperature
        self.max_new_tokens = gen_cfg.max_new_tokens
        self.timeout_s = gen_cfg.timeout_s

    def complete(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"temperature": self.temperature,
                               "num_predict": self.max_new_tokens}}
        req = urllib.request.Request(
            f"{self.base_url}/api/generate", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "")


class DashScopeBackend:
    """DashScope OpenAI 兼容端点；api_key 只来自环境变量（见 config.load_config）。"""

    def __init__(self, gen_cfg: _cfg.GenerationConfig):
        api_key = gen_cfg.dashscope.api_key
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY / OPENAI_API_KEY 环境变量")
        from openai import OpenAI

        self._client = OpenAI(base_url=gen_cfg.dashscope.base_url, api_key=api_key)
        self.model = gen_cfg.dashscope.model
        self.temperature = gen_cfg.temperature
        self.max_new_tokens = gen_cfg.max_new_tokens

    def complete(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature, max_tokens=self.max_new_tokens)
        return resp.choices[0].message.content or ""


def build_backend(cfg: _cfg.Config | None = None) -> LLMBackend:
    cfg = cfg or _cfg.get_config()
    if cfg.generation.backend == "dashscope":
        return DashScopeBackend(cfg.generation)
    return OllamaBackend(cfg.generation)


def build_prompt(query: str, evidences: Sequence[Evidence], context_summary: str = "") -> str:
    """团队模板 + [kp_id] 引用增强。资料行以 [kp_id] 前缀，便于模型就地标注。

    ``context_summary``（P6 多轮）：非空时在资料前附一段**已确认上下文摘要**（由 session 只取
    最近几轮拼成，不塞全部历史），供模型衔接追问；仍要求只依据本轮资料作答。"""
    text_block = "\n".join(f"[{e.kp_id}] {e.text}" for e in evidences)
    pre = ""
    if context_summary:
        pre = ("已确认的上下文（前几轮，仅供理解追问指代，不要直接照抄作答）：\n"
               f"{context_summary}\n\n")
    return (
        pre
        + "请只依据资料回答，资料未涵盖的内容不要编造；"
          "每个论断后用方括号标注支撑它的资料 kp_id，例如 [kp_id]。\n"
        f"资料: {text_block}\n\n问题: {query}\n请基于资料回答:"
    )


def extract_citations(text: str) -> list[str]:
    return _CITATION.findall(text)


def _citations_ok(answer: str, ev_ids: set[str], require: bool) -> bool:
    refs = extract_citations(answer)
    if not all(r in ev_ids for r in refs):
        return False
    return bool(refs) if require else True


def generate(query: str, evidences: Sequence[Evidence], *, backend: LLMBackend,
             cfg: _cfg.Config | None = None, context_summary: str = "") -> GenerationResult:
    """基于证据生成答案并校验引用；失败重试，再失败降级拒答。"""
    cfg = cfg or _cfg.get_config()
    ev_ids = {e.kp_id for e in evidences}
    require = cfg.generation.citation.require_in_evidence
    prompt = build_prompt(query, evidences, context_summary)
    attempts = 0
    for _ in range(cfg.generation.citation.max_retry + 1):
        attempts += 1
        answer = backend.complete(prompt).strip()
        if answer and _citations_ok(answer, ev_ids, require):
            return GenerationResult(answer=answer, ok=True, attempts=attempts)
    return GenerationResult(answer=REFUSE_FALLBACK, ok=False, degraded=True, attempts=attempts)
