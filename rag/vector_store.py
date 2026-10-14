"""向量库：FAISS（可选加速）+ numpy 精确检索兜底，带磁盘持久化。

设计取舍（面试可讲）：
* 语料在「千条以内」时，numpy 归一化矩阵做内积 = 精确余弦检索，
  毫秒级、零依赖 —— 因此 numpy 是**兜底实现，永远可用**；
* 装了 ``faiss-cpu`` 时自动改用 FAISS ``IndexFlatIP``（同一份向量），
  为语料上万条时的近似检索留好路径；
* 磁盘格式只有一份（``vectors.npy`` + ``docs.jsonl`` + ``meta.json``），
  FAISS 只是内存加速 —— 加载时按需重建，避免双份格式不同步；
* 「离线建索引、在线只加载」：启动时读盘（快），不会发网络请求重建。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from config.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - 仅类型标注
    from rag.retriever import Document

log = get_logger("rag.vector_store")

try:  # FAISS 可选：装了用 FAISS 加速，没装走 numpy（功能一致）
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001 - 缺库/平台不兼容都当没有
    faiss = None  # type: ignore[assignment]
    _HAS_FAISS = False


def has_faiss() -> bool:
    """当前环境是否可用 FAISS（演示/日志用）。"""
    return _HAS_FAISS


def corpus_fingerprint(docs: list["Document"]) -> str:
    """语料指纹：文本集合一变，指纹就变（判断索引是否过期）。

    按 ``doc_id`` 排序后计算，与目录遍历顺序无关 —— 否则重启后
    rglob 顺序变化会被误判成「语料变了」而要求重建。
    """
    digest = hashlib.md5()
    for doc in sorted(docs, key=lambda d: d.doc_id):
        digest.update(doc.doc_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(doc.text.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


@dataclass
class IndexMeta:
    """索引元信息（随磁盘一起持久化）。"""

    model: str          # 建索引用的 embedding 模型名（换模型必须重建）
    dim: int            # 向量维度
    count: int          # 文本块数
    fingerprint: str    # 语料指纹
    created_at: float
    backend: str        # faiss / numpy（仅记录，加载时按当前环境重选）

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dim": self.dim,
            "count": self.count,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "backend": self.backend,
        }


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorIndex:
    """向量索引：归一化向量 + 余弦（内积）检索。"""

    def __init__(self, docs: list["Document"], matrix: np.ndarray, meta: IndexMeta) -> None:
        self.docs = docs
        self.meta = meta
        self._matrix = _l2_normalize(matrix.astype(np.float32))
        self._faiss_index = None
        if _HAS_FAISS:
            self._faiss_index = faiss.IndexFlatIP(self._matrix.shape[1])
            self._faiss_index.add(self._matrix)

    # ---------------------------------------------------------------- 构建
    @classmethod
    def build(cls, vectors: list[list[float]], docs: list["Document"], *, model: str) -> "VectorIndex":
        """由向量列表 + 文档列表构建索引。"""
        if len(vectors) != len(docs):
            raise ValueError(f"向量数({len(vectors)})与文档数({len(docs)})不一致")
        if not vectors:
            raise ValueError("向量列表为空：语料目录没有可索引的文本")
        matrix = np.asarray(vectors, dtype=np.float32)
        meta = IndexMeta(
            model=model,
            dim=int(matrix.shape[1]),
            count=len(docs),
            fingerprint=corpus_fingerprint(docs),
            created_at=time.time(),
            backend="faiss" if _HAS_FAISS else "numpy",
        )
        log.info("vector index built: {} docs, dim={}, backend={}", len(docs), meta.dim, meta.backend)
        return cls(docs, matrix, meta)

    # ---------------------------------------------------------------- 检索
    def search(self, query_vector: list[float], top_k: int = 4) -> list[tuple[float, "Document"]]:
        """返回 (相似度分数, 文档) 列表，按分数降序；分数 <= 0 的不返回。"""
        if not self.docs or top_k <= 0:
            return []
        query = np.asarray([query_vector], dtype=np.float32)
        query = _l2_normalize(query)
        k = min(top_k, len(self.docs))
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query, k)
            pairs = [
                (float(scores[0][i]), self.docs[int(indices[0][i])])
                for i in range(k)
                if int(indices[0][i]) >= 0
            ]
        else:
            sims = (self._matrix @ query[0]).tolist()
            order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
            pairs = [(float(sims[i]), self.docs[i]) for i in order]
        return [(score, doc) for score, doc in pairs if score > 0.0]

    # ---------------------------------------------------------------- 持久化
    def save(self, directory: str | Path) -> Path:
        """把索引落盘（npy + jsonl + meta.json）。"""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "vectors.npy", self._matrix)
        with (root / "docs.jsonl").open("w", encoding="utf-8") as fp:
            for doc in self.docs:
                fp.write(json.dumps(
                    {"doc_id": doc.doc_id, "text": doc.text, "meta": doc.meta},
                    ensure_ascii=False,
                ) + "\n")
        (root / "meta.json").write_text(
            json.dumps(self.meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        log.info("vector index saved: {} ({} docs)", root, len(self.docs))
        return root

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex | None":
        """从磁盘加载索引；缺失或损坏时返回 None（上层退回词法检索）。"""
        from rag.retriever import Document  # 延迟导入，避免循环依赖

        root = Path(directory)
        npy_path, docs_path, meta_path = root / "vectors.npy", root / "docs.jsonl", root / "meta.json"
        if not (npy_path.exists() and docs_path.exists() and meta_path.exists()):
            return None
        try:
            matrix = np.load(npy_path)
            docs = [
                Document(doc_id=item["doc_id"], text=item["text"], meta=item.get("meta") or {})
                for item in (json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines() if line.strip())
            ]
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = IndexMeta(
                model=str(raw.get("model", "")),
                dim=int(raw.get("dim", matrix.shape[1])),
                count=int(raw.get("count", len(docs))),
                fingerprint=str(raw.get("fingerprint", "")),
                created_at=float(raw.get("created_at", 0.0)),
                backend=str(raw.get("backend", "numpy")),
            )
        except Exception as err:  # noqa: BLE001 - 索引坏了就走词法，不阻断启动
            log.warning("vector index load failed ({}): {}", root, err)
            return None
        if len(docs) != matrix.shape[0]:
            log.warning("vector index inconsistent: {} docs vs {} vectors", len(docs), matrix.shape[0])
            return None
        return cls(docs, matrix, meta)

    def matches(self, *, model: str, fingerprint: str) -> bool:
        """索引是否仍然有效：模型一致 + 语料未变。"""
        return self.meta.model == model and self.meta.fingerprint == fingerprint
