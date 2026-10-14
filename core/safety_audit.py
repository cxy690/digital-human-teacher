"""安全事件审计：把危机/敏感命中落盘，供人工复核与事后追溯。

**为什么必须落盘，而不是只推一个 WebSocket 事件**：
``session.push_event("safety_alert", ...)`` 只到「当前这条连接」——
没人在线就没有人收到，页面一刷新就没了，事后也查不到。
危机干预要求**可追溯、可复盘、可追责**，所以事件必须持久化。

存储形态选 **JSONL**（一行一条）而不是单个 JSON 文件：
安全事件是只追加的时间序列，JSONL 天然支持 append 与流式读取，
不会像整份 JSON 那样每次写入都得重写全文，也不会因为一条损坏而全盘不可读。

写入是同步的（与 :class:`core.checkpoint.Checkpointer` 保持一致的取舍）：
一次追加只有一行，代价远小于把每个事件都丢进线程池的复杂度。
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from config.logging import get_logger
from config.settings import settings

log = get_logger("core.safety_audit")


class SafetyAudit:
    """安全事件审计日志（只追加）。"""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path if path is not None else settings.safety_audit_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, payload: dict) -> dict | None:
        """追加一条安全事件，返回落盘后的记录；关闭或失败时返回 ``None``。

        **失败必须静默**：审计写不进去也不能影响正在进行的答疑与危机响应。
        """
        if not settings.safety_audit_enable:
            return None
        entry = {"ts": time.time(), "ts_iso": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
        try:
            line = json.dumps(entry, ensure_ascii=False)
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except (OSError, TypeError) as err:
            log.warning("安全事件落盘失败（不影响主链路）：{}", err)
            return None
        log.info("安全事件已落盘：{}", self.path)
        return entry

    def recent(self, limit: int = 50) -> list[dict]:
        """最近若干条，**倒序**（最新在前）；读失败一律返回空列表。

        当前实现每次全量读取再取尾部——事件量级极小（只有危机/敏感命中才写），
        这个取舍换来的是「无需索引、文件被手工截断也不会崩」。
        事件量真上来了再换 SQLite / 轮转文件。
        """
        if limit <= 0 or not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as err:
            log.warning("安全事件读取失败：{}", err)
            return []
        out: list[dict] = []
        for raw in reversed(lines[-limit:]):
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                # 单行损坏（如写了一半被 kill）不应让整份日志不可读
                log.debug("跳过损坏的安全事件行：{}", raw[:60])
        return out


#: 进程内共享单例（与 ``core.state.store.session_store``、``core.a2a.bus.a2a_bus`` 同一用法）：
#: 共用一份文件与同一把锁，避免各会话各写一份导致交错。
safety_audit = SafetyAudit()


__all__ = ["SafetyAudit", "safety_audit"]
