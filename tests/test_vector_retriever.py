"""向量检索链路测试：索引构建/检索/持久化、后端选择与降级。

全部用 FakeEmbedder（确定性向量），不依赖任何真实 Embedding 服务。
"""
from __future__ import annotations

import pytest

from config.settings import settings
from rag import retriever as retriever_mod
from rag.embedding import EmbeddingUnavailable
from rag.retriever import Document, Retriever, VectorRetriever, build_retriever
from rag.vector_store import VectorIndex, corpus_fingerprint


def _vec(text: str) -> list[float]:
    """确定性伪向量：话题各占一个正交方向（用于验证排序）。"""
    if "面积" in text or "三角" in text:
        return [1.0, 0.0, 0.0]
    if "历史" in text or "清朝" in text:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


class FakeEmbedder:
    """假编码器：可注入失败，记录调用。"""

    model_name = "fake-embed"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise EmbeddingUnavailable("boom")
        self.calls.append(list(texts))
        return [_vec(t) for t in texts]


def _docs() -> list[Document]:
    return [
        Document(doc_id="a:0", text="三角形面积等于底乘高除以2", meta={"source": "textbook"}),
        Document(doc_id="b:0", text="清朝历史知识点", meta={"source": "curriculum"}),
        Document(doc_id="c:0", text="英语语法时态", meta={"source": "textbook"}),
    ]


# ---------------------------------------------------------------- 向量库
def test_index_build_search_order() -> None:
    docs = _docs()
    index = VectorIndex.build([_vec(d.text) for d in docs], docs, model="fake-embed")
    # 手工向量：与 a（面积）方向最接近、b（历史）次之 —— 验证排序而非编码
    hits = index.search([1.0, 0.35, 0.0], top_k=2)
    assert [doc.doc_id for _, doc in hits] == ["a:0", "b:0"]
    assert hits[0][0] > hits[1][0]
    # 纯「面积」话题：只命中 a（其余分数为 0 被过滤），且分数接近 1
    only = index.search(_vec("三角形的面积"), top_k=2)
    assert [doc.doc_id for _, doc in only] == ["a:0"]
    assert only[0][0] == pytest.approx(1.0, abs=1e-5)


def test_index_save_load_roundtrip(tmp_path) -> None:
    docs = _docs()
    index = VectorIndex.build([_vec(d.text) for d in docs], docs, model="fake-embed")
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded is not None
    assert loaded.matches(model="fake-embed", fingerprint=corpus_fingerprint(docs))
    hits = loaded.search(_vec("清朝历史"), top_k=1)
    assert hits and hits[0][1].text == "清朝历史知识点"


def test_index_load_missing_or_damaged(tmp_path) -> None:
    assert VectorIndex.load(tmp_path) is None  # 目录为空
    (tmp_path / "vectors.npy").write_bytes(b"not-npy")
    (tmp_path / "docs.jsonl").write_text('{"doc_id":"x","text":"y"}\n', encoding="utf-8")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    assert VectorIndex.load(tmp_path) is None  # 损坏 -> None（上层退词法）


# ---------------------------------------------------------------- 检索器
async def test_vector_retriever_returns_hits() -> None:
    docs = _docs()
    embedder = FakeEmbedder()
    index = VectorIndex.build([_vec(d.text) for d in docs], docs, model="fake-embed")
    r = VectorRetriever(embedder, index, fallback=Retriever(docs))
    out = await r.retrieve("三角形的面积", top_k=1)
    assert out == ["三角形面积等于底乘高除以2"]


async def test_vector_retriever_falls_back_on_error() -> None:
    docs = _docs()
    embedder = FakeEmbedder(fail=True)
    index = VectorIndex.build([_vec(d.text) for d in docs], docs, model="fake-embed")
    r = VectorRetriever(embedder, index, fallback=Retriever(docs))
    out = await r.retrieve("三角形的面积", top_k=1)
    # 编码失败 -> 当次查询降级词法，仍能命中（bigram 有「三角」「面积」重叠）
    assert out and "三角" in out[0]


async def test_vector_retriever_error_without_fallback() -> None:
    docs = _docs()
    index = VectorIndex.build([_vec(d.text) for d in docs], docs, model="fake-embed")
    r = VectorRetriever(FakeEmbedder(fail=True), index, fallback=None)
    assert await r.retrieve("三角形的面积") == []


# ---------------------------------------------------------------- 后端选择
def test_build_retriever_keyword_without_embedding_config(monkeypatch) -> None:
    monkeypatch.setattr(settings, "embedding_api_base", "")
    monkeypatch.setattr(settings, "rag_backend", "auto")
    monkeypatch.setattr(retriever_mod, "load_documents", _docs)
    r = build_retriever()
    assert isinstance(r, Retriever)
    assert r.backend_name


def test_build_retriever_vector_when_index_ready(monkeypatch, tmp_path) -> None:
    docs = _docs()
    monkeypatch.setattr(settings, "embedding_api_base", "http://fake/v1")
    monkeypatch.setattr(settings, "rag_backend", "auto")
    monkeypatch.setattr(settings, "rag_index_dir", str(tmp_path))
    monkeypatch.setattr(retriever_mod, "load_documents", lambda: docs)
    monkeypatch.setattr(retriever_mod, "build_embedding_client", FakeEmbedder)
    VectorIndex.build([_vec(d.text) for d in docs], docs, model="fake-embed").save(tmp_path)
    r = build_retriever()
    assert isinstance(r, VectorRetriever)
    assert isinstance(r.fallback, Retriever)


def test_build_retriever_stale_index_falls_back(monkeypatch, tmp_path) -> None:
    docs = _docs()
    monkeypatch.setattr(settings, "embedding_api_base", "http://fake/v1")
    monkeypatch.setattr(settings, "rag_backend", "vector")
    monkeypatch.setattr(settings, "rag_index_dir", str(tmp_path))
    # 索引用旧语料建：指纹与当前 docs 不一致 -> 判定过期 -> 退回词法
    monkeypatch.setattr(retriever_mod, "load_documents", lambda: docs + [
        Document(doc_id="d:0", text="新增的语料", meta={}),
    ])
    monkeypatch.setattr(retriever_mod, "build_embedding_client", FakeEmbedder)
    VectorIndex.build([_vec(d.text) for d in docs], docs, model="fake-embed").save(tmp_path)
    assert isinstance(build_retriever(), Retriever)
