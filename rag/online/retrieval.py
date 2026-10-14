# -*- coding: utf-8 -*-
"""混合检索候选生成（P1 dense 基线 + P3 稀疏路）。

链路第 3 步：产出**两路**候选，交给 rerank.py 做 RRF 融合与阈值收口（README 第 4 步）。
  A. 稠密路：bge-m3 向量（COSINE，现有）——三集合 qa_semantic / kp_semantic / 分块集合。
  B. 稀疏路：
       · 分块集合（kp_chunks / kp_chunks_v2）—— Milvus `sparse` 字段（服务端 BM25 Function，
         承载文本 `text`，中文分词），传原始 query 即混合检索。
       · qa / kp 路——**本地 rank-bm25 兜底**（语料 = MySQL qa_item.canonical_question /
         kp.search_text，进程内建索引，在线只读）。kp_semantic 无 sparse 字段，本地 bm25 是其
         唯一词面信号；bm25 命中但稠密未召回的 kp 作为互补候选注入。

每个候选带 `dense_rank/dense_score` 与 `sparse_rank/sparse_score`（1 起，0=该路未命中），
`.score` 先置为稠密余弦（未经精排时的退化序），融合终分由 rerank.py 覆写。

**status 标量预过滤——真实取值（2026-09 describe+抽样实测，勿臆造）**：
  qa_semantic approved(25)；kp_semantic approved(12)/candidate(336)；
  kp_chunks approved(26)/candidate(389)；允许集合默认 {approved, candidate}。
  qa.text_kind ∈ {canonical,paraphrase,negative}：negative 不作答案命中。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import config as _cfg
from . import embedder, store


@dataclass
class QaHit:
    qa_id: str
    score: float
    text_kind: str
    kp_id: str
    policy: str
    in_scope: bool
    student_intent: str
    question_text: str
    item: dict[str, Any] = field(default_factory=dict)   # MySQL 水合后的 qa_item 全行
    dense_rank: int = 0
    sparse_rank: int = 0
    dense_score: float = 0.0
    sparse_score: float = 0.0


@dataclass
class ChunkHit:
    chunk_id: str
    kp_id: str
    text: str
    chunk_type: str
    page_no: int
    slide_no: int
    score: float
    parent_id: str = ""         # 仅 kp_chunks_v2：子块回指父块（节）的 chunk_id
    chunk_strategy: str = ""    # 仅 kp_chunks_v2：default/semantic/parent_child
    parent_text: str = ""       # 仅 v2+parent_child：小-到大展开的父块正文（检索后只读回捞）
    dense_rank: int = 0
    sparse_rank: int = 0
    dense_score: float = 0.0
    sparse_score: float = 0.0


@dataclass
class KpHit:
    kp_id: str
    text: str
    text_kind: str
    score: float
    dense_rank: int = 0
    sparse_rank: int = 0
    dense_score: float = 0.0
    sparse_score: float = 0.0


@dataclass
class RetrievalResult:
    qa_hits: list[QaHit]
    kp_hits: list[KpHit]
    chunk_hits: list[ChunkHit]
    matched_by: str           # exact|paraphrase|vector|no_match

    @property
    def top_qa(self) -> QaHit | None:
        return self.qa_hits[0] if self.qa_hits else None


def _status_filter(field_name: str, allowed: list[str]) -> str:
    vals = ", ".join(f'"{v}"' for v in allowed)
    return f"{field_name} in [{vals}]"


def _search(client: Any, collection: str, data: list[Any], anns_field: str, expr: str,
            output_fields: list[str], limit: int) -> list[dict[str, Any]]:
    """单路向量检索：稠密传向量、稀疏传原始文本（Milvus BM25 Function）；拍平为实体 dict。"""
    res = client.search(collection_name=collection, data=data, anns_field=anns_field,
                        filter=expr, limit=limit, output_fields=output_fields)
    hits = res[0] if res else []
    out = []
    for h in hits:
        ent = h.get("entity", {})
        ent["_score"] = float(h.get("distance", 0.0))
        out.append(ent)
    return out


# ---------------------------------------------------------------------------
# 本地 rank-bm25 兜底（只读 MySQL，进程内建索引并缓存；jieba 切中文词）
# ---------------------------------------------------------------------------
_BM25: dict[str, tuple[Any, list[str], dict[str, str]]] = {}

_BM25_SQL = {
    "kp": "SELECT kp_id, search_text FROM kp WHERE search_text IS NOT NULL AND search_text <> ''",
    "qa": "SELECT qa_id, canonical_question AS search_text FROM qa_item "
          "WHERE canonical_question IS NOT NULL AND canonical_question <> ''",
}
_BM25_ID = {"kp": "kp_id", "qa": "qa_id"}


def _tokenize(text: str) -> list[str]:
    import jieba  # 延迟导入：仅在真正建/查本地索引时用到

    return [t for t in jieba.lcut((text or "").lower()) if t.strip()]


def _bm25_index(name: str):
    """惰性构建并缓存 (BM25Okapi, ids, text_by_id)；语料缺失/依赖缺失时返回 None（降级不影响稠密路）。"""
    if name in _BM25:
        return _BM25[name]
    try:
        from rank_bm25 import BM25Okapi
    except Exception:  # noqa: BLE001 - 未安装即优雅降级为纯稠密
        return None
    id_col = _BM25_ID[name]
    try:
        rows = store.query(_BM25_SQL[name])
    except Exception:  # noqa: BLE001 - MySQL 抖动不阻断检索
        return None
    ids, texts, by_id = [], [], {}
    for r in rows:
        kid = str(r.get(id_col) or "")
        txt = str(r.get("search_text") or "")
        if not kid or not txt.strip():
            continue
        ids.append(kid)
        texts.append(_tokenize(txt))
        by_id[kid] = txt
    if not texts or not any(texts):
        return None
    _BM25[name] = (BM25Okapi(texts), ids, by_id)
    return _BM25[name]


def _bm25_rank(name: str, query: str, top_k: int) -> dict[str, tuple[int, float]]:
    """返回 {doc_id: (rank, raw_bm25)}（仅 score>0，按分降序取前 top_k）。"""
    idx = _bm25_index(name)
    if idx is None:
        return {}
    bm, ids, _ = idx
    qtok = _tokenize(query)
    if not qtok:
        return {}
    scores = bm.get_scores(qtok)
    order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    out: dict[str, tuple[int, float]] = {}
    rank = 0
    for i in order:
        if scores[i] <= 0:
            break
        rank += 1
        out[ids[i]] = (rank, float(scores[i]))
        if rank >= top_k:
            break
    return out


def _bm25_text(name: str, doc_id: str) -> str:
    idx = _bm25_index(name)
    return idx[2].get(doc_id, "") if idx else ""


def reset_for_tests() -> None:
    """释放进程内 bm25 索引缓存（不参与生产路径）。"""
    _BM25.clear()


# ---------------------------------------------------------------------------
# 水合
# ---------------------------------------------------------------------------
def _hydrate_qa(hits: list[QaHit]) -> None:
    ids = [h.qa_id for h in hits if h.qa_id]
    if not ids:
        return
    ph = store.placeholders(ids)
    rows = store.query(f"SELECT * FROM qa_item WHERE qa_id IN ({ph})", ids)
    by_id = {r["qa_id"]: r for r in rows}
    for h in hits:
        h.item = by_id.get(h.qa_id, {})


def _chunks_is_v2(name: str) -> bool:
    """非旧 kp_chunks 即视为带 parent_id/chunk_strategy 的版本化集合（如 kp_chunks_v2）。"""
    return name != "kp_chunks"


def _hydrate_parents(client: Any, collection: str, hits: list[ChunkHit]) -> None:
    """小-到大：对 parent_child 子块，按 parent_id 只读回捞父块（节）正文。"""
    pids = sorted({h.parent_id for h in hits if h.parent_id})
    if not pids or not hasattr(client, "query"):
        return
    vals = ", ".join(f'"{p}"' for p in pids)
    try:
        rows = client.query(collection_name=collection, filter=f"chunk_id in [{vals}]",
                            output_fields=["chunk_id", "text"])
    except Exception:  # noqa: BLE001 - 父块展开是锦上添花，不能因此打断主链
        return
    by_id = {r.get("chunk_id", ""): r.get("text", "") for r in rows or []}
    for h in hits:
        if h.parent_id:
            h.parent_text = by_id.get(h.parent_id, "")


def _matched_by(qa_hits: list[QaHit], other_hits: list[Any], threshold: float) -> str:
    """稠密锚定的初始归属（终归属由 rerank.py 在融合后重算）。"""
    if qa_hits and max(h.dense_score for h in qa_hits) >= threshold:
        top = max(qa_hits, key=lambda h: h.dense_score)
        return "exact" if top.text_kind == "canonical" else "paraphrase"
    if other_hits and max(getattr(h, "dense_score", 0.0) for h in other_hits) >= threshold:
        return "vector"
    return "no_match"


def retrieve(query: str, *, client: Any | None = None, cfg: _cfg.Config | None = None
             ) -> RetrievalResult:
    cfg = cfg or _cfg.get_config()
    client = client or store.milvus()
    coll = cfg.milvus.collections
    rc = cfg.retrieval
    vec = embedder.embed_one(query)
    hybrid = rc.hybrid_enabled

    st = _status_filter("status", rc.allowed_status)
    qa_kinds = _status_filter("text_kind", rc.qa_text_kinds)

    # ---- A. QA 路：稠密 + 本地 bm25(canonical_question) 重排（不注入纯词面候选：fixed 答案须语义确认）----
    raw_qa = _search(client, coll["qa"], [vec], "dense",
                     f"{st} && {qa_kinds} && in_scope == true",
                     ["qa_id", "kp_id", "text_kind", "policy", "in_scope",
                      "student_intent", "question_text", "status"], rc.top_k)
    qa_sr = _bm25_rank("qa", query, rc.sparse_top_k) if (hybrid and rc.local_bm25) else {}
    qa_hits: list[QaHit] = []
    for i, r in enumerate(raw_qa):
        if not r.get("qa_id"):
            continue
        srank, sscore = qa_sr.get(r["qa_id"], (0, 0.0))
        qa_hits.append(QaHit(qa_id=r.get("qa_id", ""), score=r["_score"],
                             text_kind=r.get("text_kind", ""), kp_id=r.get("kp_id", ""),
                             policy=r.get("policy", ""), in_scope=bool(r.get("in_scope", False)),
                             student_intent=r.get("student_intent", ""),
                             question_text=r.get("question_text", ""),
                             dense_rank=i + 1, dense_score=r["_score"],
                             sparse_rank=srank, sparse_score=sscore))

    # ---- B. KP 路：稠密 + 本地 bm25(kp.search_text)；bm25-only 的 kp 作互补候选注入 ----
    raw_kp = _search(client, coll["kp"], [vec], "dense", st,
                     ["kp_id", "text", "text_kind", "status"], rc.top_k)
    kp_sr = _bm25_rank("kp", query, rc.sparse_top_k) if (hybrid and rc.local_bm25) else {}
    kp_hits: list[KpHit] = []
    seen_kp: set[str] = set()
    for i, r in enumerate(raw_kp):
        kid = r.get("kp_id", "")
        if not kid:
            continue
        srank, sscore = kp_sr.get(kid, (0, 0.0))
        kp_hits.append(KpHit(kp_id=kid, text=r.get("text", ""),
                             text_kind=r.get("text_kind", ""), score=r["_score"],
                             dense_rank=i + 1, dense_score=r["_score"],
                             sparse_rank=srank, sparse_score=sscore))
        seen_kp.add(kid)
    for kid, (srank, sscore) in sorted(kp_sr.items(), key=lambda kv: kv[1][0]):
        if kid in seen_kp:
            continue
        kp_hits.append(KpHit(kp_id=kid, text=_bm25_text("kp", kid), text_kind="bm25",
                             score=0.0, sparse_rank=srank, sparse_score=sscore))

    # ---- C. 分块路：稠密 + Milvus sparse（服务端 BM25），按 chunk_id 合并两路 ----
    chunks_coll = rc.chunks_collection or coll["chunks"]
    chunk_fields = ["chunk_id", "kp_id", "text", "chunk_type", "page_no", "slide_no", "status"]
    chunk_filter = st
    if _chunks_is_v2(chunks_coll):
        chunk_fields += ["parent_id", "chunk_strategy"]
        chunk_filter = f'{chunk_filter} && chunk_type != "parent_section"'
        if rc.chunks_strategies:
            allow = ", ".join(f'"{s}"' for s in rc.chunks_strategies)
            chunk_filter = f"{chunk_filter} && chunk_strategy in [{allow}]"
    raw_ch = _search(client, chunks_coll, [vec], "dense", chunk_filter, chunk_fields, rc.top_k)
    raw_ch_sparse: list[dict[str, Any]] = []
    if hybrid:
        try:  # 服务端 BM25：传原始 query，anns_field=sparse
            raw_ch_sparse = _search(client, chunks_coll, [query], "sparse",
                                    chunk_filter, chunk_fields, rc.sparse_top_k)
        except Exception:  # noqa: BLE001 - 集合无 sparse / 不支持时降级为纯稠密，不阻断
            raw_ch_sparse = []

    by_cid: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(raw_ch):
        cid = r.get("chunk_id", "")
        if not cid:
            continue
        r["_drank"], r["_srank"] = i + 1, 0
        r["_dscore"], r["_sscore"] = r["_score"], 0.0
        by_cid[cid] = r
    for j, r in enumerate(raw_ch_sparse):
        cid = r.get("chunk_id", "")
        if not cid:
            continue
        cur = by_cid.get(cid)
        if cur is None:
            r["_drank"], r["_dscore"] = 0, 0.0
            r["_srank"], r["_sscore"] = j + 1, r["_score"]
            by_cid[cid] = r
        else:
            cur["_srank"], cur["_sscore"] = j + 1, r["_score"]

    chunk_hits: list[ChunkHit] = []
    for r in by_cid.values():
        chunk_hits.append(ChunkHit(chunk_id=r.get("chunk_id", ""), kp_id=r.get("kp_id", ""),
                                   text=r.get("text", ""), chunk_type=r.get("chunk_type", ""),
                                   page_no=int(r.get("page_no", 0) or 0),
                                   slide_no=int(r.get("slide_no", 0) or 0),
                                   score=r.get("_dscore", 0.0),
                                   parent_id=r.get("parent_id", "") or "",
                                   chunk_strategy=r.get("chunk_strategy", "") or "",
                                   dense_rank=r.get("_drank", 0), dense_score=r.get("_dscore", 0.0),
                                   sparse_rank=r.get("_srank", 0), sparse_score=r.get("_sscore", 0.0)))
    if _chunks_is_v2(chunks_coll):
        _hydrate_parents(client, chunks_coll, chunk_hits)

    _hydrate_qa(qa_hits)
    # 初始归属仅作种子（终归属由 rerank.py 按 P4.5 三路阈值重算）；此处用 qa_threshold 与旧行为对齐。
    matched_by = _matched_by(qa_hits, kp_hits + chunk_hits, rc.qa_threshold)
    return RetrievalResult(qa_hits=qa_hits, kp_hits=kp_hits, chunk_hits=chunk_hits,
                           matched_by=matched_by)
