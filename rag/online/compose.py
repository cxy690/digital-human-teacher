# -*- coding: utf-8 -*-
"""话术组装（链路第 8 步）。决定统一结构里最终 ``answer`` 文本。

分发：
  fixed        → 直接用库内标准答案（``decision.answer_text``）；缺失则兜底救援话术。
  rag_grounded → 用生成结果（引用校验失败时已是降级拒答话术）。
  template / refuse → 取 utterance_template 真实话术；DB 不可用或未命中则回退静态话术。

话术按 policy→utterance_function 映射从 MySQL 读一条只读模板（best-effort，异常/空即
回退静态），保证「无 LLM key 时 template/refuse 路径完整可用」。
"""
from __future__ import annotations

from . import config as _cfg
from . import store

POLICY_FUNCTION = {
    "REJECT_AND_LOG": "UF_GUARDRAIL_HARD",
    "REDIRECT_SOFT": "UF_GUARDRAIL_SOFT",
    "EMPATHY_THEN_REDIRECT": "UF_ENCOURAGE",
    "ANSWER_TEMPLATE": "UF_TRANSITION",
    "CLARIFY_ASK_BACK": "UF_CLARIFY",
    "ACKNOWLEDGE_AND_VERIFY": "UF_CLARIFY",
    "CONFIRM_AND_EXPLAIN": "UF_PRAISE",
    "GUIDE_NOT_GIVE": "UF_HINT",
}

STATIC = {
    "UF_GUARDRAIL_HARD": "这个话题不适合在课堂里讨论，我们回到今天的学习内容好吗？",
    "UF_GUARDRAIL_SOFT": "我们先回到课堂内容，这个问题可以课后问老师。",
    "UF_ENCOURAGE": "别着急，你已经很棒了，我们一小步一小步来。",
    "UF_CLARIFY": "能再说得具体一点吗？我想更准确地帮到你。",
    "UF_TRANSITION": "我们继续来看今天的知识点吧。",
    "UF_PRAISE": "思路很棒！我们再检查一下细节。",
    "UF_HINT": "先想想这一步用到了哪个概念，试着说说看。",
}

_SQL = (
    "SELECT ut.text FROM utterance u "
    "JOIN utterance_template ut ON ut.utterance_id = u.utterance_id "
    "WHERE u.`function` = %s AND u.status IN ('approved','published','candidate') LIMIT 1"
)

_CACHE: dict[str, str] = {}


def _script(function: str, cfg) -> str:
    if cfg.compose.get("enabled", True) and function not in _CACHE:
        try:
            rows = store.query(_SQL, [function])
            _CACHE[function] = rows[0]["text"] if rows else ""
        except Exception:  # noqa: BLE001 - 话术库读不到时必须能回退，不阻断答题
            _CACHE[function] = ""
    db = _CACHE.get(function, "")
    fallback = cfg.compose.get("fallback_function", "UF_RESCUE")
    return db or STATIC.get(function) or STATIC.get(fallback) or STATIC["UF_GUARDRAIL_SOFT"]


def reset_for_tests() -> None:
    _CACHE.clear()


def compose(decision, generated: str | None = None, *, cfg: _cfg.Config | None = None) -> str:
    cfg = cfg or _cfg.get_config()
    if decision.modality == "fixed":
        return decision.answer_text or _script(
            cfg.compose.get("fallback_function", "UF_RESCUE"), cfg)
    if decision.modality == "rag_grounded":
        return generated or ""
    return _script(POLICY_FUNCTION.get(decision.policy, "UF_GUARDRAIL_SOFT"), cfg)
