"""RAG 检索：为答疑注入教材/讲义/题库上下文。

两种后端，同一接口（``async retrieve(query, top_k) -> list[str]``）：

* :class:`Retriever` —— 词法检索：字符二元组（bigram）相似度，零重依赖，
  始终可用（兜底）；
* :class:`VectorRetriever` —— 向量检索：Embedding 编码 + 向量库
  （见 :mod:`rag.embedding` / :mod:`rag.vector_store`），语义匹配更强。

:func:`build_retriever` 按配置自动选择（``RAG_BACKEND``）：配了 Embedding
服务且索引就绪就用向量，否则词法 —— 调用方（Supervisor / TaskGraph /
LangGraph / MCP）完全无感知。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import settings
from rag.embedding import EmbeddingUnavailable, build_embedding_client
from rag.vector_store import VectorIndex, corpus_fingerprint

log = get_logger("rag.retriever")

_TEXT_SUFFIXES = {".txt", ".md", ".jsonl"}
_SEARCH_DIRS = ["data/textbook", "data/teaching_materials", "data/curriculum", "data/question_bank"]


@dataclass
class Document:
    """一条可检索文本。"""

    doc_id: str
    text: str
    meta: dict[str, str] = field(default_factory=dict)


def _bigrams(text: str) -> Counter:
    cleaned = re.sub(r"\s+", "", text)
    return Counter(cleaned[i : i + 2] for i in range(max(len(cleaned) - 1, 0)))


def _similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[g] * b[g] for g in common)
    denom = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return num / denom if denom else 0.0


class Retriever:
    """轻量文本检索器（词法 bigram，兜底实现）。"""

    backend_name = "词法检索（bigram）"

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self._index = [(doc, _bigrams(doc.text)) for doc in documents]
        # 实例级后端名带块数：/health 与启动日志可直接看出词法兜底时的语料规模
        self.backend_name = f"词法检索（bigram，{len(documents)} 块）"

    async def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """返回与 query 最相关的若干文本片段。"""
        if not query.strip() or not self._index:
            return []
        q = _bigrams(query)
        scored = [(_similarity(q, grams), doc.text) for doc, grams in self._index]
        scored.sort(key=lambda x: x[0], reverse=True)
        k = top_k or settings.rag_top_k
        return [text for score, text in scored[:k] if score > 0.0]


def load_documents(dirs: list[str] | None = None) -> list[Document]:
    """从数据目录加载所有文本片段（按行切分为片段）。"""
    docs: list[Document] = []
    for d in dirs or _SEARCH_DIRS:
        root = Path(d)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in _TEXT_SUFFIXES or not path.is_file():
                continue
            try:
                for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                    line = line.strip()
                    if len(line) >= 4:
                        docs.append(Document(doc_id=f"{path.stem}:{i}", text=line, meta={"source": str(path)}))
            except OSError as err:
                log.warning("load {} failed: {}", path, err)
    log.info("loaded {} document chunks", len(docs))
    return docs


class VectorRetriever:
    """向量检索：Embedding 编码 + 向量库（FAISS / numpy）。

    运行时查询编码失败（服务抖动/超时）时，**当次查询自动降级**到词法
    检索（构造时可注入 fallback）—— 与项目其他环节一致的容错原则。
    """

    backend_name = "向量检索（embedding + 向量库）"

    def __init__(
        self,
        embedder: Any,
        index: VectorIndex,
        *,
        fallback: Retriever | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.fallback = fallback

    async def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """向量检索 top_k 片段；编码失败时退回词法（有 fallback 时）。"""
        if not query.strip():
            return []
        k = top_k or settings.rag_top_k
        try:
            vectors = await self.embedder.embed([query])
        except EmbeddingUnavailable as err:
            log.warning("query embedding failed, fallback to keyword: {}", err)
            if self.fallback is not None:
                return await self.fallback.retrieve(query, top_k=k)
            return []
        hits = self.index.search(vectors[0], top_k=k)
        return [doc.text for _, doc in hits]


def build_retriever() -> Any:
    """构建检索器：按 ``RAG_BACKEND`` 自动选择向量 / 词法。

    * ``auto``（默认）：配了 Embedding 服务且向量索引已就绪 → 向量检索；
      否则词法 bigram（与旧版行为完全一致）。
    * ``vector``：强制向量；服务未配 / 索引缺失或过期时**告警并退回词法**，
      不阻断启动（先跑 ``scripts/build_rag_index.py`` 生成索引）。
    * ``keyword``：强制词法。

    返回对象统一实现 ``async retrieve(query, top_k) -> list[str]`` 与
    ``backend_name`` 属性，调用方无感知。
    """
    backend = (settings.rag_backend or "auto").strip().lower()
    docs = load_documents()
    keyword = Retriever(docs)
    if backend == "keyword":
        return keyword

    client = build_embedding_client()
    if client is None:
        if backend == "vector":
            log.warning("RAG_BACKEND=vector 但未配置 EMBEDDING_API_BASE，退回词法检索")
        return keyword

    index = VectorIndex.load(settings.rag_index_dir)
    if index is None:
        log.warning(
            "未找到向量索引（先运行 python scripts/build_rag_index.py 生成），本次退回词法检索"
        )
        return keyword
    if not index.matches(model=client.model_name, fingerprint=corpus_fingerprint(docs)):
        log.warning("向量索引已过期（语料或模型发生变化），请重新运行 build_rag_index.py；本次退回词法检索")
        return keyword

    log.info(
        "RAG 后端：向量检索（{} 块，索引后端 {}，模型 {}）",
        len(index.docs),
        index.meta.backend,
        index.meta.model,
    )
    retriever = VectorRetriever(client, index, fallback=keyword)
    # 实例级 backend_name 带上块数与模型：/health 与启动日志据此确认「向量检索真的生效」
    retriever.backend_name = f"向量检索（{len(index.docs)} 块，模型 {index.meta.model}）"
    return retriever
