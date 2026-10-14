# -*- coding: utf-8 -*-
"""在线链路编排器（链路第 9 步）。串起 改写→检索→重排→上下文→策略→生成→话术，分段计时。

产出契约里的统一结构：
  {answer, modality, citations[], policy, latency_ms{total,retrieve,generate},
   matched_by, rewrite{original,rewritten,level,changed,drift,cosine}, session_id}
（P4 新增 rewrite 日志字段；citations 仍固定四键 kp_id/evidence/page_no/score 不变。）
**P6 多轮**：显式传入 ``session_id`` 且 ``session.enabled`` 时，按会话载入近 N 轮→主题切换检测
（bge 余弦过低即重置上下文，不换 id）→历史注入改写（代词消解）→已确认上下文摘要入生成→写回
Redis（不可用降级内存）；此时额外返回 ``session{turn_no,store,topic_switch,context_turns}``。
不传 session_id（或关会话）时与单轮完全一致：不读写会话存储、不做主题检测编码。
无 LLM key 时 fixed/refuse/template 全程可用；rag_grounded 若后端不可用则降级为拒答话术。
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

from . import (compose, config as _cfg, context, generation, policy, query_rewrite,
               rerank, retrieval, session)


def ask(query: str, *, cfg: _cfg.Config | None = None, client: Any | None = None,
        backend: generation.LLMBackend | None = None,
        session_id: str | None = None) -> dict[str, Any]:
    cfg = cfg or _cfg.get_config()
    t0 = perf_counter()

    # ── P6 多轮：仅当显式传入 session_id 且开启会话时才接管存储；单轮路径与此前完全一致。──
    use_session = bool(session_id) and cfg.session.enabled
    eff_turns: list[dict] = []
    turn_no, store, topic_switch = 1, "none", False
    if use_session:
        turns, _ = session.load(session_id, cfg=cfg)
        turn_no = len(turns) + 1
        last_q = session.last_question(turns)
        # 主题一致性：本轮与上一问 bge 余弦过低 → 判换主题 → 重置上下文（沿用同一 session_id）。
        if last_q:
            sim = query_rewrite.semantic_similarity(query, last_q)
            topic_switch = sim is not None and sim < cfg.session.topic_reset_threshold
        eff_turns = [] if topic_switch else turns

    history = session.history_questions(eff_turns) or None   # 供 LLM 级指代消解/补全
    rr = query_rewrite.rewrite(query, history=history, cfg=cfg, backend=backend)
    q = rr.rewritten                             # 检索/重排用改写后的独立 query
    tr = perf_counter()
    result = rerank.rerank(retrieval.retrieve(q, client=client, cfg=cfg), cfg=cfg, query=q)
    retrieve_ms = (perf_counter() - tr) * 1000

    ctx = context.build_context(result, cfg=cfg)
    decision = policy.decide(query, result, cfg=cfg)   # 策略判定以**原 query**为准

    # 第 2 轮起（且未换主题）给生成附一段"已确认上下文"摘要，不塞全部历史以控 token。
    context_summary = session.context_summary(eff_turns, cfg=cfg) if eff_turns else ""

    generate_ms = 0.0
    generated: str | None = None
    modality = decision.modality
    if decision.needs_generation:
        tg = perf_counter()
        try:
            gr = generation.generate(query, ctx.evidences,
                                     backend=backend or generation.build_backend(cfg), cfg=cfg,
                                     context_summary=context_summary)
            generated = gr.answer
            if gr.degraded:
                modality = "refuse"
        except Exception:  # noqa: BLE001 - 后端不可用时必须降级，不能让整链崩
            generated, modality = generation.REFUSE_FALLBACK, "refuse"
        generate_ms = (perf_counter() - tg) * 1000

    answer = compose.compose(decision, generated, cfg=cfg)
    total_ms = (perf_counter() - t0) * 1000
    # 拒答不基于证据作答，不应回显引用（避免误导"这是有据的回答"）。
    citations = [] if modality == "refuse" else ctx.citations

    out: dict[str, Any] = {
        "answer": answer,
        "modality": modality,
        "citations": citations,
        "policy": decision.policy,
        "latency_ms": {"total": round(total_ms), "retrieve": round(retrieve_ms),
                       "generate": round(generate_ms)},
        "matched_by": result.matched_by,
        "rewrite": rr.as_dict(),
        "session_id": session_id or cfg.session_id_default,
    }

    if use_session:
        # 写回本轮（citations 存精简形，控 Redis 体积）；返回实际所用存储后端。
        store = session.append(session_id, {
            "question": query, "rewritten": rr.rewritten, "answer": answer,
            "citations": [{"kp_id": c["kp_id"], "page_no": c.get("page_no"),
                           "score": c.get("score"), "evidence": (c.get("evidence") or "")[:120]}
                          for c in citations]}, cfg=cfg)
        out["session"] = {"turn_no": turn_no, "store": store,
                          "topic_switch": topic_switch, "context_turns": len(eff_turns)}
    return out
