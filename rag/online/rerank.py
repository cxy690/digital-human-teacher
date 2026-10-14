# -*- coding: utf-8 -*-
"""RRF 融合 + 阈值收口 +（可选）精排（链路第 4 步，P3）。

输入是 retrieval.py 产出的两路候选（每个命中带 dense_rank/dense_score 与
sparse_rank/sparse_score）。本模块负责：

  1. **RRF 融合**：`rrf = Σ_路 1/(k + rank)`（k=`retrieval.rrf_k`，默认 60），在候选池内
     归一到 [0,1] 作为最终 `.score`（context.py 按 `.score` 降序取用，返回结构不变）。
  2. **阈值收口（P4.5 分档）**：稠密余弦 ≥ **本路阈值**（qa=`qa_threshold` / kp=`kp_threshold` /
     chunks=`chunks_threshold`，三路独立可配）即语义确认；仅被稀疏路命中（稠密未过本路阈值）的
     候选，需 BM25 归一分（`s/(s+1)`）≥ `hybrid_floor` 才允许进入上下文（防词面噪声劫持）。
     qa 路更严：只有稠密过阈才计 exact/paraphrase（fixed 答案必须语义确认，词面不单独触发）。
  3. **（可选）精排**：`retrieval.rerank.enabled` 且 bge-reranker 可加载时对证据路做交叉编码
     重排；**无 GPU / 未装依赖时保持关闭，RRF 序即为最终序**（接口留好、优雅降级）。

设计说明：跨三路（含本地 bm25）的融合用自实现 RRF，而非 Milvus 原生 `hybrid_search`
（后者只在分块集合内融合 dense+sparse，无法跨 qa/kp/chunks 三路与本地兜底统一融合）；
Milvus 原生 hybrid_search 的能力在交付实测中单独验证。
"""
from __future__ import annotations

from typing import Any, Callable

from . import config as _cfg

_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(msg)


def _rrf(dense_rank: int, sparse_rank: int, k: int) -> float:
    v = 0.0
    if dense_rank:
        v += 1.0 / (k + dense_rank)
    if sparse_rank:
        v += 1.0 / (k + sparse_rank)
    return v


def _squash(raw: float) -> float:
    """BM25 原始分压到 [0,1)（单调、有界），使服务端 BM25 与本地 bm25 可共用同一 hybrid_floor。"""
    return raw / (raw + 1.0) if raw > 0 else 0.0


def _prep(hits: list[Any]) -> None:
    """兼容手搭结果（如离线单测直接构造命中）：无任何路信息时，把 `.score` 视作稠密分。

    生产路径里 retrieve() 必给 dense_rank/sparse_rank，本函数不改动它们；仅为不破坏
    既有“只填 score”的调用方提供退化兼容。
    """
    for h in hits:
        if not h.dense_rank and not h.sparse_rank and not h.dense_score and getattr(h, "score", 0.0):
            h.dense_score = float(h.score)
            h.dense_rank = 1


def _fuse(hits: list[Any], *, k: int, rescue_ok: bool, dense_thr: float,
          floor: float) -> list[Any]:
    """融合打分→归一→收口→按融合分降序。rescue_ok=False 时纯稠密门（qa 路）。"""
    if not hits:
        return hits
    raw = [_rrf(h.dense_rank, h.sparse_rank, k) for h in hits]
    mx = max(raw) or 1.0
    kept: list[Any] = []
    for h, r in zip(hits, raw):
        sem_ok = h.dense_score >= dense_thr
        lex_ok = rescue_ok and h.sparse_rank > 0 and _squash(h.sparse_score) >= floor
        if not (sem_ok or lex_ok):
            continue
        h.score = round(r / mx, 4)
        kept.append(h)
    kept.sort(key=lambda h: h.score, reverse=True)
    return kept


def _matched_by(qa_hits: list[Any], kp_hits: list[Any], chunk_hits: list[Any],
                qa_thr: float, kp_thr: float, chunks_thr: float) -> str:
    """exact/paraphrase/vector 一律以稠密语义确认（各按**本路阈值**）为准。

    纯词面（BM25）救援可把候选带入上下文以补召回，但在**无任何稠密过阈命中**时不得把
    no_match 翻成 vector——否则超纲问题会因跨学科词面噪声（光/化学/历史题命中数学语料的
    公共字）被判为「已匹配」而走 ANSWER_FROM_RAG，击穿拒答门（与模块 docstring 的语义
    确认口径一致）。dense-only 模式下池内命中必已过稠密阈，此判定与旧行为等价。
    P4.5 分档：qa 用 qa_thr、kp 用 kp_thr、chunks 用 chunks_thr，qa 误命中代价高故阀值最高。
    """
    confirmed = [q for q in qa_hits if q.dense_score >= qa_thr]
    if confirmed:
        top = max(confirmed, key=lambda q: q.dense_score)
        return "exact" if top.text_kind == "canonical" else "paraphrase"
    if (any(h.dense_score >= kp_thr for h in kp_hits)
            or any(h.dense_score >= chunks_thr for h in chunk_hits)):
        return "vector"
    return "no_match"


def _gate_dense_only(result: Any, rc: _cfg.RetrievalConfig) -> Any:
    """hybrid_enabled=False：退回纯稠密（等价 V1/P2 门控），但仍按 P4.5 三路各自阈值。"""
    for lst, thr in ((result.qa_hits, rc.qa_threshold),
                     (result.kp_hits, rc.kp_threshold),
                     (result.chunk_hits, rc.chunks_threshold)):
        _prep(lst)
        lst[:] = sorted([h for h in lst if h.dense_score >= thr],
                        key=lambda h: h.dense_score, reverse=True)
        for h in lst:
            h.score = round(h.dense_score, 4)
    result.matched_by = _matched_by(result.qa_hits, result.kp_hits, result.chunk_hits,
                                    rc.qa_threshold, rc.kp_threshold, rc.chunks_threshold)
    return result


# ---- 可选精排：接口预留，默认关闭；不可用时保持 RRF 序 ----
_RERANKER: Callable[[str, list[str]], list[float]] | None = None
_RERANK_LOADED = False


def _load_reranker(cfg: _cfg.Config):
    """惰性加载 bge-reranker（FlagEmbedding）。未启用/缺依赖/缺 GPU 一律返回 None（保持 RRF 序）。"""
    global _RERANKER, _RERANK_LOADED
    if _RERANK_LOADED:
        return _RERANKER
    _RERANK_LOADED = True
    rr = cfg.retrieval.rerank
    if not rr.enabled or not rr.model_path:
        return None
    try:
        from FlagEmbedding import FlagReranker

        model = FlagReranker(rr.model_path, use_fp16=False)

        def _score(query: str, docs: list[str]) -> list[float]:
            return [float(x) for x in model.compute_score([[query, d] for d in docs],
                                                          normalize=True)]

        _RERANKER = _score
    except Exception as e:  # noqa: BLE001 - 精排不可用绝不能打断主链，退回 RRF 序
        _warn_once("rerank_off", f"[rerank] 精排不可用，保持 RRF 序：{type(e).__name__}: {e}")
        _RERANKER = None
    return _RERANKER


def _maybe_rerank(result: Any, cfg: _cfg.Config) -> None:
    scorer = _load_reranker(cfg)
    if scorer is None:
        return  # 关闭/不可用：RRF 序即最终序
    rr = cfg.retrieval.rerank
    pool = result.kp_hits + result.chunk_hits
    top = pool[:rr.top_n]
    docs = [(getattr(h, "parent_text", "") or h.text or "") for h in top]
    if not docs:
        return
    try:
        scores = scorer(cfg._p3_query, docs)  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        _warn_once("rerank_fail", f"[rerank] 精排打分失败，保持 RRF 序：{e}")
        return
    for h, s in zip(top, scores):
        h.score = round(s, 4)
    # 依精排分对证据路重排（qa 路不参与精排，保持 exact 判定稳定）。
    result.chunk_hits.sort(key=lambda h: h.score, reverse=True)
    result.kp_hits.sort(key=lambda h: h.score, reverse=True)


def rerank(result, *, cfg: _cfg.Config | None = None, query: str = ""):
    """RRF 融合 + 阈值收口 +（可选）精排，原地写回命中顺序与 matched_by。"""
    cfg = cfg or _cfg.get_config()
    rc = cfg.retrieval
    if not rc.hybrid_enabled:
        return _gate_dense_only(result, rc)
    _prep(result.qa_hits)
    _prep(result.kp_hits)
    _prep(result.chunk_hits)
    result.qa_hits = _fuse(result.qa_hits, k=rc.rrf_k, rescue_ok=False,
                           dense_thr=rc.qa_threshold, floor=rc.hybrid_floor)
    result.kp_hits = _fuse(result.kp_hits, k=rc.rrf_k, rescue_ok=True,
                           dense_thr=rc.kp_threshold, floor=rc.hybrid_floor)
    result.chunk_hits = _fuse(result.chunk_hits, k=rc.rrf_k, rescue_ok=True,
                              dense_thr=rc.chunks_threshold, floor=rc.hybrid_floor)
    if query:
        cfg._p3_query = query  # 供可选精排读取原始 query（不新增返回字段，仅进程内传递）
    _maybe_rerank(result, cfg)
    result.matched_by = _matched_by(result.qa_hits, result.kp_hits, result.chunk_hits,
                                    rc.qa_threshold, rc.kp_threshold, rc.chunks_threshold)
    return result
