"""构建 RAG 向量索引（离线跑一次；应用启动时只加载，不在线重建）。

用法（项目根执行）：
    python scripts/build_rag_index.py                        # 增量：语料/模型未变则跳过
    python scripts/build_rag_index.py --rebuild              # 强制重建
    python scripts/build_rag_index.py --query 三角形的面积     # 建完顺便试检索一条

前置：.env 配置 Embedding 服务（OpenAI 兼容 /v1/embeddings）：
    EMBEDDING_API_BASE=http://127.0.0.1:8101/v1     # 云端私有化（vLLM / Infinity / TEI）或第三方 API
    EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
    EMBEDDING_API_KEY=                              # 视服务而定，可留空

产物：data/raw/rag_index/（vectors.npy + docs.jsonl + meta.json）
     —— 语料或模型变化后需重跑；应用侧检测到过期会自动退回词法检索并告警。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # 语料走 data/ 相对路径：切到项目根，保证任意 cwd 下都能跑
try:  # Windows 控制台中文输出
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config.settings import settings  # noqa: E402
from rag.embedding import EmbeddingUnavailable, build_embedding_client  # noqa: E402
from rag.retriever import load_documents  # noqa: E402
from rag.vector_store import VectorIndex, corpus_fingerprint, has_faiss  # noqa: E402


async def build(*, rebuild: bool, query: str | None) -> int:
    """构建（或复用）向量索引；返回进程退出码。"""
    docs = load_documents()
    if not docs:
        print("[x] 语料为空：请先往 data/textbook / curriculum / question_bank 放文本")
        return 1
    client = build_embedding_client()
    if client is None:
        print("[x] 未配置 Embedding 服务：请在 .env 设置 EMBEDDING_API_BASE（OpenAI 兼容 /v1/embeddings）")
        print("    例：EMBEDDING_API_BASE=http://127.0.0.1:8101/v1   EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5")
        return 1

    fingerprint = corpus_fingerprint(docs)
    if not rebuild:
        current = VectorIndex.load(settings.rag_index_dir)
        if current is not None and current.matches(model=client.model_name, fingerprint=fingerprint):
            print(f"[=] 索引已是最新（{len(current.docs)} 块，模型 {current.meta.model}），跳过。"
                  f"如需强制重建加 --rebuild")
            return 0

    print(f"[1] 语料：{len(docs)} 个文本块（指纹 {fingerprint[:12]}…）")
    print(f"[2] 编码：模型 {client.model_name} @ {client.api_base}"
          f"（批大小 {client.batch_size}，向量库后端 {'faiss' if has_faiss() else 'numpy'}）")
    texts = [d.text for d in docs]
    vectors: list[list[float]] = []
    started = time.time()
    batch = client.batch_size
    try:
        for i in range(0, len(texts), batch):
            vectors.extend(await client.embed(texts[i : i + batch]))
            print(f"    {min(i + batch, len(texts))}/{len(texts)}")
    except EmbeddingUnavailable as err:
        print(f"[x] 编码失败：{err}")
        print("    检查 EMBEDDING_API_BASE 服务是否在线、模型名是否正确、是否需要 API Key")
        return 1
    cost = time.time() - started

    index = VectorIndex.build(vectors, docs, model=client.model_name)
    index.save(settings.rag_index_dir)
    print(f"[3] 索引已保存：{settings.rag_index_dir}"
          f"（{index.meta.count} 块 · 维度 {index.meta.dim} · 用时 {cost:.1f}s）")

    if query:
        vec = (await client.embed([query]))[0]
        hits = index.search(vec, top_k=settings.rag_top_k)
        print(f"[4] 试检索 {query!r} → {len(hits)} 条：")
        for i, (score, doc) in enumerate(hits, 1):
            print(f"    {score:.3f}  {doc.text[:64]}")
            print(f"           {doc.meta.get('source', '?')}")
    print("\n完成：应用重启后自动切换到向量检索（RAG_BACKEND=auto 即可）")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument("--rebuild", action="store_true", help="强制重建（忽略现有索引）")
    parser.add_argument("--query", default="", help="建完后试检索一条查询，用于自检")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(build(rebuild=args.rebuild, query=args.query or None)))


if __name__ == "__main__":
    main()
