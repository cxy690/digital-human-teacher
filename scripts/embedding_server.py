# -*- coding: utf-8 -*-
"""本地 OpenAI 兼容 Embedding 服务（供项目 RAG 向量检索使用）。

为什么自部署：学生提问不出内网、零 API 费用；项目 ``rag/embedding.py`` 走的是
OpenAI 兼容协议（``POST {base}/embeddings``），本服务就是把这个协议在云端本地实现。

依赖（装在独立环境 ``.venvs/embed``，避免污染主应用 venv）：
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install sentence-transformers

启动（云端）：``bash deploy/cloud/restart_embedding.sh``
手动：``python scripts/embedding_server.py --port 8101 --model BAAI/bge-small-zh-v1.5``

接口：
    GET  /health          -> {"status": "ok", "model": ..., "dim": 512}
    POST /v1/embeddings   -> OpenAI 兼容：{"model": "...", "input": "文本" 或 ["文本", ...]}
                             返回 {"data": [{"index": i, "embedding": [...]}, ...], ...}

说明：bge v1.5 系列官方建议**无需指令前缀**，直接编码即可；输出向量已做 L2
归一化（项目侧还会再归一化一次，无害）。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # Windows 控制台中文输出
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_LOCK = threading.Lock()   # CPU 编码串行化，避免并发抢占
_MODEL = None              # SentenceTransformer，由 main() 加载
_MODEL_NAME = ""
_DIM = 0


def _encode(texts: list[str]) -> list[list[float]]:
    with _LOCK:
        vectors = _MODEL.encode(texts, normalize_embeddings=True, batch_size=32)
    return [list(map(float, v)) for v in vectors]


class Handler(BaseHTTPRequestHandler):
    server_version = "edu-embedding/1.0"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # 打到 stderr（由启动脚本重定向进日志文件）
        sys.stderr.write("[embed] %s - %s\n" % (self.address_string(), fmt % args))

    # ---------------- GET
    def do_GET(self) -> None:  # noqa: N802 - http.server 命名约定
        path = self.path.rstrip("/")
        if path in ("/health", "/healthz"):
            self._send(200, {"status": "ok", "model": _MODEL_NAME, "dim": _DIM})
        elif path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": _MODEL_NAME, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    # ---------------- POST
    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/embeddings":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            raw = payload.get("input")
            if isinstance(raw, str):
                texts = [raw]
            elif isinstance(raw, list) and all(isinstance(x, str) for x in raw):
                texts = list(raw)
            else:
                self._send(400, {"error": "input 必须是字符串或字符串列表"})
                return
            if not texts:
                self._send(400, {"error": "input 不能为空"})
                return
            started = time.time()
            vectors = _encode(texts)
            cost = time.time() - started
            data = [
                {"object": "embedding", "index": i, "embedding": vec}
                for i, vec in enumerate(vectors)
            ]
            self._send(200, {"object": "list", "data": data, "model": _MODEL_NAME,
                             "usage": {"prompt_tokens": sum(len(t) for t in texts), "total_tokens": 0}})
            sys.stderr.write(f"[embed] {len(texts)} 条编码完成，用时 {cost:.2f}s\n")
        except Exception as err:  # noqa: BLE001 - 统一 500，交给调用方（项目侧会降级词法）
            self._send(500, {"error": f"embedding 失败：{err}"})


def main() -> None:
    global _MODEL, _MODEL_NAME, _DIM
    parser = argparse.ArgumentParser(description="本地 OpenAI 兼容 embedding 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--device", default="cpu", help="cpu / cuda（默认 cpu，避免与数字人抢显存）")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer  # 延迟导入：--help 不触发重依赖

    print(f"[embed] 加载模型 {args.model}（device={args.device}）…", flush=True)
    started = time.time()
    _MODEL = SentenceTransformer(args.model, device=args.device)
    _MODEL_NAME = args.model
    # 兼容新旧 sentence-transformers：新版本改名为 get_embedding_dimension
    _get_dim = getattr(_MODEL, "get_embedding_dimension", None) or _MODEL.get_sentence_embedding_dimension
    _DIM = int(_get_dim() or 0)
    print(f"[embed] 模型就绪：dim={_DIM}，用时 {time.time() - started:.1f}s", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[embed] 服务启动：http://{args.host}:{args.port}（POST /v1/embeddings）", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
