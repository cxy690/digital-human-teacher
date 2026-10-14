# -*- coding: utf-8 -*-
"""上下文组装（链路第 5 步）。把检索命中合并去重为「证据集」并登记引用。

优先用 kp 证据块（chunk_hits，带页码），不足再补知识点摘要（kp_hits）；按分数降序、
按 (kp_id,text) 去重，截到 ``retrieval.final_k``。产出的 citations 直接对齐统一返回结构。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import config as _cfg
from . import generation


@dataclass
class Context:
    evidences: list[generation.Evidence] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)


def build_context(result, *, cfg: _cfg.Config | None = None) -> Context:
    cfg = cfg or _cfg.get_config()
    limit = cfg.retrieval.final_k
    ranked = sorted(result.chunk_hits, key=lambda c: c.score, reverse=True) + \
        sorted(result.kp_hits, key=lambda k: k.score, reverse=True)

    ctx = Context()
    seen: set[tuple[str, str]] = set()
    for hit in ranked:
        kp_id = hit.kp_id
        # 小-到大：v2 parent_child 子块带回 parent_text 时，用更大的父块（节）作上下文；
        # 否则用命中自身文本。引用仍按命中的 kp_id/score/page_no，返回结构不变。
        text = (getattr(hit, "parent_text", "") or hit.text or "").strip()
        key = (kp_id, text[:64])
        if not text or key in seen:
            continue
        seen.add(key)
        ctx.evidences.append(generation.Evidence(kp_id=kp_id, text=text))
        ctx.citations.append({
            "kp_id": kp_id, "evidence": text,
            "page_no": getattr(hit, "page_no", None), "score": round(hit.score, 4)})
        if len(ctx.evidences) >= limit:
            break
    return ctx
