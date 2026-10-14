# -*- coding: utf-8 -*-
"""rag.online —— 在线 RAG 检索增强问答运行时（对三存储只读）。

链路顺序（契约见 rag/online/README.md）::

    query_rewrite → embedder → retrieval → rerank → context
    → policy → generation → compose → pipeline/cli

铁律：本包只读 MySQL / Milvus / Neo4j；任何写库都属于离线链路。
"""
