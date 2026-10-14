# -*- coding: utf-8 -*-
"""多轮会话存储（P6，session.py）。

链路里第 0/末步：请求进来时按 ``session_id`` 载入近 N 轮问答（供改写注入历史、生成附带
上下文摘要），作答后把本轮写回。契约见 rag/online/README.md（``rag:session:{id}``）。

- **Redis 优先**：key=``session.key_prefix + session_id``（默认 ``rag:session:``），值=JSON
  数组 ``[{question, rewritten, answer, citations}, ...]``，写时 ``SETEX`` 带 TTL（默认 30min）。
  为守"零新依赖"（与 generation.py 用标准库 urllib 调 Ollama 同口径），这里用**标准库 socket
  实现的极简 RESP 客户端**，不引 ``redis`` 包。
- **降级**：Redis 连接/读写任一异常 → 落**进程内存 dict**（同样按 TTL 语义）并**告警一次**，
  绝不阻断主链。
- 只读三存储铁律不涉及本模块：会话是运行态数据、非教研资产，Redis 与 MySQL/Milvus/Neo4j 无关。
"""
from __future__ import annotations

import json
import logging
import socket
import time
from urllib.parse import urlparse

from . import config as _cfg

_LOG = logging.getLogger("rag.online.session")

# 进程内存兜底：key -> (expires_at, json_str)。Redis 不可用时使用。
_MEMORY: dict[str, tuple[float, str]] = {}
_WARNED = False


def _warn_once(exc: BaseException) -> None:
    global _WARNED
    if not _WARNED:
        _WARNED = True
        _LOG.warning("会话 Redis 不可用，降级进程内存 dict（首次原因：%s: %s）",
                     type(exc).__name__, exc)


def _parse_url(url: str) -> tuple[str, int, int]:
    p = urlparse(url)
    db = 0
    if p.path and p.path.strip("/"):
        try:
            db = int(p.path.strip("/"))
        except ValueError:
            db = 0
    return p.hostname or "127.0.0.1", int(p.port or 6379), db


class MiniRedis:
    """极简 RESP 客户端（仅 GET / SETEX / DEL / PING），只用标准库 socket。"""

    def __init__(self, url: str, timeout: float = 1.0):
        self.host, self.port, self.db = _parse_url(url)
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._rf = None

    def _ensure(self) -> None:
        if self._sock is None:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self._rf = self._sock.makefile("rb")
            if self.db:
                self._cmd("SELECT", str(self.db))

    def _cmd(self, *args) -> object:
        self._ensure()
        buf = b"*%d\r\n" % len(args)
        for a in args:
            b = a if isinstance(a, bytes) else str(a).encode("utf-8")
            buf += b"$%d\r\n" % len(b) + b + b"\r\n"
        try:
            self._sock.sendall(buf)
            return self._read()
        except Exception:
            self.close()  # 连接不可信，下次重建
            raise

    def _read(self) -> object:
        line = self._rf.readline()
        if not line:
            raise ConnectionError("redis 连接被对端关闭")
        t, rest = line[:1], line[1:].rstrip(b"\r\n")
        if t == b"+":
            return rest.decode("utf-8")
        if t == b"-":
            raise RuntimeError(f"redis 返回错误：{rest.decode('utf-8', 'replace')}")
        if t == b":":
            return int(rest)
        if t == b"$":
            n = int(rest)
            if n == -1:
                return None
            data = self._rf.read(n)
            self._rf.read(2)  # 结尾 \r\n
            return data.decode("utf-8")
        raise RuntimeError(f"未知 RESP 类型：{t!r}")

    def get(self, key: str) -> str | None:
        return self._cmd("GET", key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._cmd("SETEX", key, int(ttl), value)

    def delete(self, key: str) -> None:
        self._cmd("DEL", key)

    def close(self) -> None:
        for attr in ("_rf", "_sock"):
            obj = getattr(self, attr, None)
            try:
                if obj is not None:
                    obj.close()
            except Exception:  # noqa: BLE001
                pass
        self._rf = self._sock = None


_R: MiniRedis | None = None


def _redis(cfg) -> MiniRedis:
    global _R
    if _R is None:
        _R = MiniRedis(cfg.redis_url)
    return _R


# ── 内存兜底的 TTL 语义 ──
def _mem_get(key: str) -> str | None:
    item = _MEMORY.get(key)
    if not item:
        return None
    expires_at, val = item
    if time.time() >= expires_at:
        _MEMORY.pop(key, None)
        return None
    return val


def _mem_set(key: str, val: str, ttl: int) -> None:
    _MEMORY[key] = (time.time() + ttl, val)


def _key(session_id: str, cfg) -> str:
    return cfg.session.key_prefix + session_id


def load(session_id: str, *, cfg=None) -> tuple[list[dict], str]:
    """读回某会话近 N 轮问答，返回 ``(turns, backend)``；backend ∈ {redis, memory}。

    无 session、会话不存在或已过期 → ``([], backend)``。读取失败（如 Redis 挂了）降级内存。
    """
    cfg = cfg or _cfg.get_config()
    if not session_id:
        return [], "none"
    key = _key(session_id, cfg)
    try:
        raw = _redis(cfg).get(key)
        backend = "redis"
    except Exception as e:  # noqa: BLE001 - 存储故障绝不打断问答，降级内存
        _warn_once(e)
        raw = _mem_get(key)
        backend = "memory"
    return (json.loads(raw) if raw else []), backend


def append(session_id: str, turn: dict, *, cfg=None) -> str:
    """追加本轮问答并保留最近 ``max_turns`` 轮，写回存储（带 TTL）；返回实际所用 backend。"""
    cfg = cfg or _cfg.get_config()
    if not session_id:
        return "none"
    key = _key(session_id, cfg)
    turns, _ = load(session_id, cfg=cfg)
    turns.append(turn)
    turns = turns[-cfg.session.max_turns:]
    payload = json.dumps(turns, ensure_ascii=False)
    try:
        _redis(cfg).setex(key, cfg.session.ttl_seconds, payload)
        return "redis"
    except Exception as e:  # noqa: BLE001
        _warn_once(e)
        _mem_set(key, payload, cfg.session.ttl_seconds)
        return "memory"


# ── 供 pipeline 组装历史/摘要的便捷函数 ──
def history_questions(turns: list[dict]) -> list[str]:
    """改写用的历史：历轮**原始问题**（不含本轮）。"""
    return [t["question"] for t in turns if t.get("question")]


def last_question(turns: list[dict]) -> str | None:
    return turns[-1].get("question") if turns else None


def context_summary(turns: list[dict], *, cfg=None) -> str:
    """生成用的"已确认上下文"摘要：只取最近 ``context_summary_turns`` 轮，控制 token。"""
    cfg = cfg or _cfg.get_config()
    if not turns:
        return ""
    lines = []
    for t in turns[-cfg.session.context_summary_turns:]:
        q = (t.get("question") or "").strip()
        a = (t.get("answer") or "").strip()
        if len(a) > 120:
            a = a[:120] + "…"
        if q:
            lines.append(f"- 上一问：{q}\n  已确认作答：{a}")
    return "\n".join(lines)


def reset_for_tests() -> None:
    """测试收尾：关连接、清内存与告警位（不参与生产路径）。"""
    global _R, _WARNED
    if _R is not None:
        _R.close()
    _R = None
    _MEMORY.clear()
    _WARNED = False
