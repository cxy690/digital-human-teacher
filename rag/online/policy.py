# -*- coding: utf-8 -*-
"""策略路由（P1，链路第 6 步）。

职责（README 第 6 步 + docs/16）：把「风险 / 意图 / 命中范围」映射到唯一处置，决定
下游是否进 LLM。**风险内容与 fixed 答案都不经过生成模型**——前者省延迟且 LLM 可被绕过，
后者库内已有标准答案。判定顺序对齐 safety_policy_matrix 的 decision_order（V1 取前三步）：

  1. 风险关键词命中（config.policy.risk_keywords）→ refuse + 上报，绝不进 LLM。
  2. 命中库内 in_scope 的 qa：按其 answer_modality 分发
       fixed   → 直接返回 qa.answer_text（needs_generation=False）
       refuse/template → 交 compose 出话术，不进生成模型
       rag_grounded    → 需基于证据生成
  3. 无命中 / 仅向量证据 / 超纲：
       有向量证据 → rag_grounded（开放题，用 kp/chunks 依据作答）
       仅意图关键词 → 按矩阵 intent_policy 分发（情绪/闲聊走 template 话术）
       其余（超纲、无命中、矩阵缺该 intent）→ 默认 refuse；缺 intent 时打印升级警告。

矩阵来自 MySQL ``safety_policy_matrix``（在线只读，全表进内存），可按 intent/risk 查。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import config as _cfg
from . import store

# 矩阵 policy 码 → 下游模态。未列出的策略默认 refuse（宁可不答也不误答）。
POLICY_MODALITY = {
    "ANSWER_FROM_RAG": "rag_grounded",
    "GUIDE_NOT_GIVE": "rag_grounded",
    "GENERATE_WITHIN_KP": "rag_grounded",
    "CONFIRM_AND_EXPLAIN": "rag_grounded",
    "ACKNOWLEDGE_AND_VERIFY": "rag_grounded",
    "EMPATHY_THEN_REDIRECT": "template",
    "ANSWER_TEMPLATE": "template",
    "REDIRECT_SOFT": "template",
    "CLARIFY_ASK_BACK": "template",
    "REJECT_AND_LOG": "refuse",
}

_GENERATION_POLICIES = {"GUIDE_NOT_GIVE", "GENERATE_WITHIN_KP",
                        "CONFIRM_AND_EXPLAIN", "ACKNOWLEDGE_AND_VERIFY"}


@dataclass
class PolicyDecision:
    modality: str                 # fixed | rag_grounded | template | refuse
    policy: str                   # 命中的策略码（ANSWER_FROM_RAG / REJECT_AND_LOG ...）
    intent: str | None = None
    needs_generation: bool = False    # 仅 rag_grounded 为 True
    answer_text: str | None = None    # fixed 时为库内标准答案，直接返回
    need_human_log: bool = False
    escalate: bool = False
    max_answer_chars: int | None = None


@dataclass
class Matrix:
    by_intent: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_risk: dict[str, dict[str, Any]] = field(default_factory=dict)


_MATRIX: Matrix | None = None

_MATRIX_SQL = (
    "SELECT student_intent, risk_category, policy, need_human_log, escalate, "
    "max_answer_chars FROM safety_policy_matrix WHERE enabled = 1"
)


def load_matrix(*, rows: list[dict[str, Any]] | None = None, reload: bool = False) -> Matrix:
    """把安全策略矩阵读入内存（按 intent / risk 双索引）；结果进程内缓存。"""
    global _MATRIX
    if _MATRIX is not None and not reload and rows is None:
        return _MATRIX
    data = rows if rows is not None else store.query(_MATRIX_SQL)
    m = Matrix()
    for r in data:
        if r.get("student_intent"):
            m.by_intent[r["student_intent"]] = r
        if r.get("risk_category"):
            m.by_risk[r["risk_category"]] = r
    _MATRIX = m
    return m


def reset_for_tests() -> None:
    global _MATRIX
    _MATRIX = None


def _kw_hit(query: str, mapping: dict[str, Any]) -> str | None:
    for key, words in (mapping or {}).items():
        if any(w in query for w in (words or [])):
            return key
    return None


def _in_scope(qa) -> bool:
    if "in_scope" in (qa.item or {}):
        return bool(qa.item.get("in_scope"))
    return bool(qa.in_scope)


def _chars(row: dict[str, Any] | None) -> int | None:
    val = (row or {}).get("max_answer_chars")
    return int(val) if val else None


def decide(query: str, result, *, cfg: _cfg.Config | None = None,
           matrix: Matrix | None = None) -> PolicyDecision:
    """给定原问题与检索结果，产出唯一的策略决策。"""
    cfg = cfg or _cfg.get_config()
    matrix = matrix or load_matrix()
    pol = cfg.policy or {}

    # 1) 风险前置：命中即拒答上报，不过 LLM。
    if _kw_hit(query, pol.get("risk_keywords")):
        row = matrix.by_risk.get("SR_SELF_HARM") or {}
        return PolicyDecision("refuse", row.get("policy", "REJECT_AND_LOG"),
                              need_human_log=True, escalate=True,
                              max_answer_chars=_chars(row))

    qa = result.top_qa
    intent = (qa.student_intent if qa and qa.student_intent
              else _kw_hit(query, pol.get("intent_keywords")))

    # 2) 命中库内 in_scope 的 qa：按其 answer_modality 分发。
    if qa and qa.item and _in_scope(qa):
        modality = qa.item.get("answer_modality")
        row = matrix.by_intent.get(intent) or {}
        policy = qa.policy or row.get("policy") or "ANSWER_FROM_RAG"
        if modality == "fixed":
            return PolicyDecision("fixed", policy, intent,
                                  answer_text=qa.item.get("answer_text"),
                                  max_answer_chars=_chars(row))
        if modality in ("refuse", "template"):
            return PolicyDecision(modality, policy, intent,
                                  need_human_log=bool(row.get("need_human_log")),
                                  escalate=bool(row.get("escalate")),
                                  max_answer_chars=_chars(row))
        # rag_grounded（或无模态但要求引用）：有证据 → 生成。
        if result.matched_by in ("exact", "paraphrase", "vector"):
            return PolicyDecision("rag_grounded", policy, intent, needs_generation=True,
                                  max_answer_chars=_chars(row))

    # 3) 无库内答案：区分开放题（有向量证据）与超纲/无命中。
    if result.matched_by == "vector":
        row = matrix.by_intent.get(intent or "SI_CONCEPT") or {}
        policy = row.get("policy", "ANSWER_FROM_RAG")
        modality = POLICY_MODALITY.get(policy, "rag_grounded")
        return PolicyDecision(modality, policy, intent,
                              needs_generation=(modality == "rag_grounded"),
                              max_answer_chars=_chars(row))

    if intent:
        row = matrix.by_intent.get(intent)
        if row is None:
            print(f"[policy][升级警告] 矩阵缺少 intent={intent} 的策略，按拒答兜底")
            return PolicyDecision("refuse", "REJECT_AND_LOG", intent,
                                  need_human_log=True, escalate=True)
        modality = POLICY_MODALITY.get(row["policy"], "refuse")
        return PolicyDecision(modality, row["policy"], intent,
                              needs_generation=bool(
                                  row["policy"] in _GENERATION_POLICIES
                                  and result.chunk_hits),
                              need_human_log=bool(row.get("need_human_log")),
                              escalate=bool(row.get("escalate")),
                              max_answer_chars=_chars(row))

    # 超纲 / 无命中 / qa 存在但 out-of-scope → 默认拒答。
    return PolicyDecision("refuse", "REDIRECT_SOFT", None)
