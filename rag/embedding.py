"""Embedding 客户端：OpenAI 兼容 /v1/embeddings 协议。

向量检索的「编码器」：把文本块与查询语句编码成向量，交给
:mod:`rag.vector_store` 做相似度检索。协议与 :mod:`models.llm` 一致
（OpenAI 兼容），因此云端私有化（vLLM / Infinity / TEI / Ollama 等）
与第三方 API 都能直接接入。

**没配服务时返回 None，检索自动退回词法 bigram** —— 与项目其余部分
（LLM / ASR / TTS / 数字人）相同的「可插拔 + 自动降级」原则。
"""
from __future__ import annotations

from typing import Protocol

import httpx

from config.logging import get_logger
from config.settings import settings

log = get_logger("rag.embedding")


class EmbeddingUnavailable(RuntimeError):
    """Embedding 服务不可用（网络 / 鉴权 / 超时 / 协议异常）。"""


class Embedder(Protocol):
    """编码器协议：批量文本 -> 向量列表（顺序与输入一致）。"""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def model_name(self) -> str: ...


class OpenAIEmbeddingClient:
    """OpenAI 兼容 embedding 客户端（POST ``{api_base}/embeddings``）。"""

    def __init__(
        self,
        api_base: str,
        model: str,
        *,
        api_key: str = "",
        batch_size: int = 16,
        timeout_seconds: float = 30.0,
    ) -> None:
        base = api_base.strip().rstrip("/")
        # 兼容两种写法：填 ".../v1"（推荐）或直接填完整的 ".../v1/embeddings"
        self.api_base = base[: -len("/embeddings")] if base.endswith("/embeddings") else base
        self._model = model
        self.api_key = api_key
        self.batch_size = max(1, batch_size)
        self.timeout = timeout_seconds

    @property
    def model_name(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本编码为向量（内部按 ``batch_size`` 分批请求）。"""
        if not texts:
            return []
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                vectors.extend(await self._embed_batch(client, batch))
        if len(vectors) != len(texts):
            raise EmbeddingUnavailable(
                f"embedding 返回数量不匹配：请求 {len(texts)} 条、返回 {len(vectors)} 条"
            )
        return vectors

    async def _embed_batch(self, client: httpx.AsyncClient, batch: list[str]) -> list[list[float]]:
        try:
            resp = await client.post(
                f"{self.api_base}/embeddings",
                headers=self._headers(),
                json={"model": self._model, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()
            items = sorted(data["data"], key=lambda it: it.get("index", 0))
            return [list(map(float, it["embedding"])) for it in items]
        except Exception as err:  # noqa: BLE001 - 统一收敛为「不可用」，由上层降级
            raise EmbeddingUnavailable(f"embedding 请求失败（{self.api_base}）：{err}") from err


def build_embedding_client() -> OpenAIEmbeddingClient | None:
    """按配置构建编码器；未配置服务地址时返回 None（检索走词法）。"""
    base = (settings.embedding_api_base or "").strip()
    if not base:
        return None
    client = OpenAIEmbeddingClient(
        base,
        settings.embedding_model,
        api_key=settings.embedding_api_key,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
    log.info("embedding client: {} @ {}", client.model_name, client.api_base)
    return client
