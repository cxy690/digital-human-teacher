# -*- coding: utf-8 -*-
"""bge-m3 编码封装（P0）。

铁律：嵌入必须与离线入库**同模型同维度**（bge-m3, 1024 维, normalize），否则查询向量
与库里已存向量不在同一空间。这里默认复用 :data:`rag.online.config` 的
``embedding.model_path``（与 tools/store_common.py 完全一致），归一化后余弦=点积。

- 模块级单例：SentenceTransformer 只加载一次（权重 4.4GB，首次慢）。
- LRU 缓存：相同文本不重复编码。
- 维度断言：任何一条向量维度 != dim 立即报错。

可测试性：``embed`` 通过 :func:`set_encoder` 注入 fake 编码器，单测无需加载真模型。
"""
from __future__ import annotations

import functools
from typing import Protocol, Sequence

from . import config as _cfg


class Encoder(Protocol):
    def __call__(self, texts: Sequence[str]) -> list[list[float]]: ...


_DIM: int | None = None
_ENCODER: Encoder | None = None


def _default_encoder() -> Encoder:
    emb_cfg = _cfg.get_config().embedding
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(emb_cfg.model_path, device="cpu")
    got = model.get_sentence_embedding_dimension()
    if got != emb_cfg.dim:
        raise RuntimeError(f"嵌入维度不符：模型 {got} vs 配置 {emb_cfg.dim}")

    def _enc(texts: Sequence[str]) -> list[list[float]]:
        vecs = model.encode(list(texts), batch_size=32,
                            normalize_embeddings=emb_cfg.normalize,
                            convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    return _enc


def set_encoder(enc: Encoder | None, *, dim: int | None = None) -> None:
    """注入编码器（测试用 fake；传 None 复原为默认惰性加载）。"""
    global _ENCODER, _DIM
    _ENCODER = enc
    _DIM = dim
    _embed_cached.cache_clear()


@functools.lru_cache(maxsize=4096)
def _embed_cached(text: str) -> tuple[float, ...]:
    vec = _active_encoder()([text])[0]
    return tuple(vec)


def _active_encoder() -> Encoder:
    global _ENCODER, _DIM
    if _ENCODER is None:
        _ENCODER = _default_encoder()
        _DIM = _cfg.get_config().embedding.dim
    return _ENCODER


def embed(texts: Sequence[str]) -> list[list[float]]:
    """把若干文本编码为向量列表；命中缓存则复用。带 1024 维守卫。"""
    if not texts:
        return []
    dim = _DIM or _cfg.get_config().embedding.dim
    out: list[list[float]] = []
    for t in texts:
        vec = list(_embed_cached(t))
        if len(vec) != dim:
            raise RuntimeError(f"嵌入维度不符：{len(vec)} != {dim}")
        out.append(vec)
    return out


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
