"""rag 包：知识检索（词法 bigram / 向量 + 向量库，可插拔后端）。

布局（2026-10-10 重组，原 ``pipeline/`` 包整体并入 ``rag/``）：

- ``rag.retriever``    —— 检索器（词法 bigram 兜底 + 向量实现，被 ``mcp_tools`` / ``core`` 直接使用）
- ``rag.embedding``    —— Embedding 服务客户端（OpenAI 兼容 ``/v1/embeddings``；未配置即退回词法）
- ``rag.vector_store`` —— 进程内轻量向量索引（FAISS 可选加速，缺失用 numpy）
- ``rag.online``       —— 在线 RAG 检索增强问答链路（重依赖：bge-m3 / Ollama / Redis）
- ``rag.offline``      —— 离线接入流水线（解析→清洗→切块→标注→出库，重依赖：pyyaml / jsonschema）

重要：``online`` 与 ``offline`` 子包**不在 rag 包顶层自动 import**——
它们各自依赖较重（sentence_transformers、yaml、jsonschema 等），自动 import 会强制所有
``import rag`` 的调用方（如 ``mcp_tools/server.py``、``core/harness.py``）付出加载成本。
调用方请按需 ``from rag.online import pipeline`` / ``from rag.offline import common``。
"""

from __future__ import annotations

# 顶层 import 必须保持轻量，避免强迫调用方加载重型依赖
# （embedding / vector_store 的第三方依赖都是可选的，缺失时自身降级，不拖累调用方）。
from rag.embedding import (
    Embedder,
    EmbeddingUnavailable,
    OpenAIEmbeddingClient,
    build_embedding_client,
)
from rag.retriever import Document, Retriever, VectorRetriever, build_retriever, load_documents
from rag.vector_store import IndexMeta, VectorIndex, corpus_fingerprint, has_faiss

__all__ = [
    # 顶层：轻量、稳定
    "Document",
    "Embedder",
    "EmbeddingUnavailable",
    "IndexMeta",
    "OpenAIEmbeddingClient",
    "Retriever",
    "VectorIndex",
    "VectorRetriever",
    "build_embedding_client",
    "build_retriever",
    "corpus_fingerprint",
    "has_faiss",
    "load_documents",
    # 子包入口（惰性 import，详见下方 __getattr__）
    "online",
    "offline",
]

# ---- 惰性子包加载：让 ``from rag.online import ...`` 与 ``import rag.online`` 都能工作 ----
_LAZY_SUBPACKAGES = {
    "online": "rag.online",
    "offline": "rag.offline",
}


def __getattr__(name: str):
    """PEP 562 风格的惰性属性访问，让 ``import rag`` 不会被 online/offline 重型依赖拖死。

    示例::

        from rag.online import pipeline, config     # 实际加载 rag.online
        from rag.offline import common              # 实际加载 rag.offline
        import rag.online as online_rag             # 走此 __getattr__ 拿到 module
    """
    if name in _LAZY_SUBPACKAGES:
        import importlib

        mod = importlib.import_module(_LAZY_SUBPACKAGES[name])
        globals()[name] = mod  # 缓存，避免再次走 importlib
        return mod
    raise AttributeError(f"module 'rag' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
